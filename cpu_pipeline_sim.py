import sys
import re
import matplotlib.pyplot as plt

# -----------------------------
# Simple Instruction Definition
# -----------------------------
INSTR_SET = {"LOAD", "STORE", "ADD", "SUB", "MOVI", "BEQ", "HALT"}

# -----------------------------
# Assembler
# -----------------------------
def assemble(lines):
    labels = {}
    program = []
    memory = [0] * 256

    # first pass: handle labels and .DATA
    pc = 0
    for line in lines:
        line = line.split(";")[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            labels[line[:-1]] = pc
            continue
        if line.upper().startswith(".DATA"):
            _, addr, val = line.split()
            addr, val = int(addr, 0), int(val, 0)
            memory[addr] = val
            continue
        pc += 1

    # second pass: instructions
    pc = 0
    for line in lines:
        line = line.split(";")[0].strip()
        if not line or line.endswith(":") or line.upper().startswith(".DATA"):
            continue
        tokens = re.split(r"[,\s]+", line.strip())
        op = tokens[0].upper()
        args = tokens[1:]
        if op not in INSTR_SET:
            raise ValueError(f"Unknown instruction: {op}")
        # resolve labels in args
        resolved = []
        for a in args:
            if a in labels:
                resolved.append(labels[a])
            elif a.startswith("R"):
                resolved.append(a)
            else:
                resolved.append(int(a, 0))
        program.append((op, resolved))
        pc += 1

    return program, memory

# -----------------------------
# Pipeline CPU
# -----------------------------
class PipelineCPU:
    def __init__(self, program, memory):
        self.program = program
        self.memory = memory[:]
        self.reg = [0] * 8
        self.pc = 0
        self.cycles = 0
        self.done = False
        self.pipeline = {"IF": None, "ID": None, "EX": None, "MEM": None, "WB": None}
        self.history = []

    def step(self):
        self.cycles += 1

        # Write Back
        wb = self.pipeline["WB"]
        if wb:
            op, args, res = wb
            if op in {"ADD", "SUB", "MOVI", "LOAD"} and res is not None:
                dst = int(args[0][1])  # R#
                self.reg[dst] = res
        self.pipeline["WB"] = None

        # MEM
        mem = self.pipeline["MEM"]
        if mem:
            op, args, val = mem
            if op == "LOAD":
                addr = val
                res = self.memory[addr]
                self.pipeline["WB"] = (op, args, res)
            elif op == "STORE":
                dst = int(args[0][1])
                addr = val
                self.memory[addr] = self.reg[dst]
            else:
                self.pipeline["WB"] = mem
        self.pipeline["MEM"] = None

        # EX
        ex = self.pipeline["EX"]
        if ex:
            op, args = ex
            res = None
            if op == "ADD":
                r1, r2 = int(args[1][1]), int(args[2][1])
                res = self.reg[r1] + self.reg[r2]
            elif op == "SUB":
                r1, r2 = int(args[1][1]), int(args[2][1])
                res = self.reg[r1] - self.reg[r2]
            elif op == "MOVI":
                res = args[1]
            elif op == "LOAD":
                addr = args[1]
                self.pipeline["MEM"] = (op, args, addr)
            elif op == "STORE":
                addr = args[1]
                self.pipeline["MEM"] = (op, args, addr)
            elif op == "BEQ":
                r1, r2, target = int(args[0][1]), int(args[1][1]), args[2]
                if self.reg[r1] == self.reg[r2]:
                    self.pc = target
                    self.flush_pipeline()
            elif op == "HALT":
                self.done = True
            if op in {"ADD", "SUB", "MOVI"}:
                self.pipeline["MEM"] = (op, args, res)
        self.pipeline["EX"] = None

        # ID
        id_stage = self.pipeline["ID"]
        if id_stage:
            op, args = id_stage
            self.pipeline["EX"] = (op, args)
        self.pipeline["ID"] = None

        # IF
        if self.pc < len(self.program) and not self.done:
            instr = self.program[self.pc]
            self.pipeline["ID"] = instr
            self.pc += 1

        # record state
        self.history.append(self.reg[:])

    def flush_pipeline(self):
        self.pipeline = {"IF": None, "ID": None, "EX": None, "MEM": None, "WB": None}

    def run(self, max_cycles=100):
        while not self.done and self.cycles < max_cycles:
            self.step()

# -----------------------------
# Visualization
# -----------------------------
def plot_registers(history):
    history = list(zip(*history))  # transpose
    for i, reg_vals in enumerate(history):
        plt.plot(reg_vals, label=f"R{i}")
    plt.xlabel("Cycle")
    plt.ylabel("Register Value")
    plt.title("Register Values per Cycle")
    plt.legend()
    plt.savefig("registers.png")
    print("Saved registers.png")

# -----------------------------
# Main
# -----------------------------
SAMPLE_PROGRAM = """
.DATA 0 5
.DATA 1 10
        MOVI R1, 0x0
        MOVI R2, 0x1
        LOAD R3, 0
        LOAD R4, 1
        ADD R5, R3, R4
        STORE R5, 2
        HALT
"""

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = SAMPLE_PROGRAM.strip().splitlines()
    program, memory = assemble(lines)
    cpu = PipelineCPU(program, memory)
    cpu.run()

    # write mem dump
    with open("mem_dump.txt", "w") as f:
        for i, v in enumerate(cpu.memory):
            if v != 0:
                f.write(f"{i}: {v}\n")
    print("Saved mem_dump.txt")

    # plot registers
    plot_registers(cpu.history)

if __name__ == "__main__":
    main()
