"""
cpu_pipeline_sim.py

Combined simulator that includes:
 1) An assembler supporting labels, immediates (#10), hex (0xA), and a simple .DATA directive
 2) A 5-stage pipelined CPU model (IF, ID, EX, MEM, WB) with hazard detection and simple forwarding
 3) A matplotlib visualization showing register values across cycles and a memory dump written to a file

Run: python cpu_pipeline_sim.py

Outputs:
 - mem_dump.txt : full memory contents after run
 - registers.png : a plot of R0..R7 values by cycle

This is a pedagogical simulator, not cycle-accurate for every microarchitectural detail, but demonstrates
stalls, forwarding, and the typical instruction cycle progression in a pipeline.

Supported instructions (assembly form):
  MOV  Rdest, src        ; src = Rn or #imm or 0xhex
  LOAD Rdest, addr       ; addr = #imm or Rn
  STORE Rsrc, addr
  ADD  Rdest, Rsrc1, Rsrc2
  SUB  Rdest, Rsrc1, Rsrc2
  MUL  Rdest, Rsrc1, Rsrc2
  DIV  Rdest, Rsrc1, Rsrc2
  JMP  label_or_index
  BEQ  R1, R2, label
  HALT

.DATA directive (place data into memory):
  .DATA addr, value

Labels: label: on its own or before an instruction

"""

import re
import sys
from collections import deque, defaultdict

# Optional plotting libs; program still runs if matplotlib missing (prints a warning)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ---------------------- Assembler ----------------------

IMM_RE = re.compile(r"^#(-?0x[0-9A-Fa-f]+|-?\d+)$")
HEX_RE = re.compile(r"^0x[0-9A-Fa-f]+$")

class Assembler:
    def __init__(self):
        self.labels = {}
        self.lines = []
        self.instrs = []
        self.data = {}  # addr -> value

    def parse_immediate(self, tok):
        tok = tok.strip()
        m = IMM_RE.match(tok)
        if m:
            inner = m.group(1)
            if inner.startswith(('0x','-0x')):
                return int(inner, 16)
            return int(inner)
        if HEX_RE.match(tok):
            return int(tok, 16)
        # register or number
        try:
            return int(tok)
        except:
            return None

    def load(self, asm_text):
        # split lines, remove comments
        raw_lines = asm_text.splitlines()
        cleaned = []
        pc = 0
        for ln in raw_lines:
            ln = ln.split(';')[0].strip()
            if not ln:
                continue
            # .DATA handling straightforward
            if ln.upper().startswith('.DATA'):
                cleaned.append(ln)
                continue
            # label?
            if ':' in ln:
                parts = ln.split(':')
                label = parts[0].strip()
                rest = ':'.join(parts[1:]).strip()
                self.labels[label] = pc
                if rest:
                    cleaned.append(rest)
                    pc += 1
                else:
                    # label only - does not advance PC
                    continue
            else:
                cleaned.append(ln)
                pc += 1
        # second pass: parse cleaned
        for ln in cleaned:
            if ln.upper().startswith('.DATA'):
                # format: .DATA addr, value
                tokens = ln.split(None,1)
                if len(tokens) == 1:
                    continue
                body = tokens[1]
                parts = [p.strip() for p in body.split(',')]
                if len(parts) >= 2:
                    addr_tok, val_tok = parts[0], parts[1]
                    addr = self.parse_immediate(addr_tok) if addr_tok else None
                    val = self.parse_immediate(val_tok)
                    if addr is None or val is None:
                        raise ValueError(f"Invalid .DATA spec: {ln}")
                    self.data[addr] = val
                continue
            parts = [p.strip() for p in re.split(r'[,\s]+', ln) if p.strip()]
            op = parts[0].upper()
            args = parts[1:]
            self.instrs.append({'op':op, 'args': args, 'text': ln})

    def resolve_labels_and_immediates(self):
        # Replace label operands with numeric PC where appropriate
        for i,ins in enumerate(self.instrs):
            new_args = []
            for a in ins['args']:
                if a in self.labels:
                    new_args.append(str(self.labels[a]))
                else:
                    new_args.append(a)
            ins['args'] = new_args

    def assemble(self, asm_text):
        self.load(asm_text)
        self.resolve_labels_and_immediates()
        return self.instrs, self.data

# ---------------------- Pipeline CPU ----------------------

# Pipeline registers are dictionaries describing relevant fields for each stage

class PipelineCPU:
    def __init__(self, instr_mem, data_mem, mem_size=1024):
        self.instr_mem = instr_mem  # list of dicts
        self.PC = 0
        self.regs = {f'R{i}':0 for i in range(8)}
        self.mem = [0]*mem_size
        for addr,val in data_mem.items():
            if 0 <= addr < mem_size:
                self.mem[addr] = val
        self.cycle = 0
        self.halted = False
        # pipeline registers
        self.IF_ID = None
        self.ID_EX = None
        self.EX_MEM = None
        self.MEM_WB = None
        # stats for visualization
        self.reg_history = defaultdict(list)  # reg->list of values by cycle (after WB)
        self.pc_history = []
        self.stalls = 0

    # helpers
    def get_reg_val(self, token):
        token = token.strip()
        if token.startswith('R'):
            return self.regs[token]
        # immediate like #10 or 0xA or decimal
        if token.startswith('#'):
            inner = token[1:]
            if inner.startswith(('0x','-0x')):
                return int(inner,16)
            return int(inner)
        if token.startswith('0x'):
            return int(token,16)
        try:
            return int(token)
        except:
            return 0

    def set_reg(self, rname, value):
        self.regs[rname] = int(value)

    # Hazard detection: detect load-use and insert one bubble (stall)
    def detect_load_use_hazard(self, id_ins):
        # id_ins is the instruction at ID stage (parsed)
        if not id_ins or not self.ID_EX:
            return False
        prev = self.ID_EX.get('instr')
        if not prev:
            return False
        if prev['op'] == 'LOAD':
            # dest reg of LOAD
            dest = prev['args'][0]
            # if ID instruction uses that dest as source, hazard
            srcs = []
            op = id_ins['op']
            if op in ('ADD','SUB','MUL','DIV'):
                srcs = [id_ins['args'][1], id_ins['args'][2]]
            elif op in ('STORE', 'BEQ'):
                srcs = id_ins['args']
            elif op in ('MOV','LOAD'):
                if len(id_ins['args'])>=2:
                    srcs = [id_ins['args'][1]]
            if any(s==dest for s in srcs):
                return True
        return False

    # Forwarding: check EX/MEM and MEM/WB for producing regs needed in EX
    def forward_value(self, regname):
        # check EX_MEM
        if self.EX_MEM and self.EX_MEM.get('write_reg')==regname and 'alu_result' in self.EX_MEM:
            return self.EX_MEM['alu_result']
        # check MEM_WB
        if self.MEM_WB and self.MEM_WB.get('write_reg')==regname:
            if 'mem_read' in self.MEM_WB:
                return self.MEM_WB['mem_read']
            if 'alu_result' in self.MEM_WB:
                return self.MEM_WB['alu_result']
        return None

    def step(self):
        if self.halted:
            return
        self.cycle += 1
        print(f"== Cycle {self.cycle} ==")

        # -------------------- WB stage --------------------
        if self.MEM_WB:
            wb = self.MEM_WB
            if wb.get('write_reg'):
                val = None
                if 'mem_read' in wb:
                    val = wb['mem_read']
                elif 'alu_result' in wb:
                    val = wb['alu_result']
                if val is not None:
                    self.set_reg(wb['write_reg'], val)
                    print(f"WB: Wrote {wb['write_reg']} = {val}")
            if wb.get('op') == 'HALT':
                print('WB: HALT seen -> halting')
                self.halted = True

        # record reg history snapshot after WB
        for r in sorted(self.regs.keys()):
            self.reg_history[r].append(self.regs[r])
        self.pc_history.append(self.PC)

        # -------------------- MEM stage --------------------
        new_MEM_WB = None
        if self.EX_MEM:
            em = self.EX_MEM
            memwb = {'op': em.get('op'), 'write_reg': em.get('write_reg')}
            if em.get('mem_write'):
                addr = em.get('mem_addr')
                val = em.get('alu_result')
                self.mem[addr] = val
                print(f"MEM: Stored mem[{addr}] = {val}")
            if em.get('mem_read'):
                addr = em.get('mem_addr')
                memwb['mem_read'] = self.mem[addr]
                print(f"MEM: Loaded mem[{addr}] = {self.mem[addr]}")
            # pass-through alu result
            if 'alu_result' in em:
                memwb['alu_result'] = em['alu_result']
            new_MEM_WB = memwb
        else:
            new_MEM_WB = None

        # -------------------- EX stage --------------------
        new_EX_MEM = None
        if self.ID_EX:
            idex = self.ID_EX
            instr = idex.get('instr')
            if instr:
                op = instr['op']
                out = {'op':op}
                # helper to get operand value, with forwarding
                def getval(tok):
                    val = None
                    if tok.startswith('R'):
                        # check forwarding
                        f = self.forward_value(tok)
                        if f is not None:
                            return f
                        return self.regs[tok]
                    else:
                        return self.get_reg_val(tok)
                if op == 'ADD':
                    v1 = getval(instr['args'][1])
                    v2 = getval(instr['args'][2])
                    out['alu_result'] = v1 + v2
                    out['write_reg'] = instr['args'][0]
                elif op == 'SUB':
                    v1 = getval(instr['args'][1])
                    v2 = getval(instr['args'][2])
                    out['alu_result'] = v1 - v2
                    out['write_reg'] = instr['args'][0]
                elif op == 'MUL':
                    v1 = getval(instr['args'][1])
                    v2 = getval(instr['args'][2])
                    out['alu_result'] = v1 * v2
                    out['write_reg'] = instr['args'][0]
                elif op == 'DIV':
                    v1 = getval(instr['args'][1])
                    v2 = getval(instr['args'][2])
                    out['alu_result'] = 0 if v2==0 else v1//v2
                    out['write_reg'] = instr['args'][0]
                elif op == 'MOV':
                    v = getval(instr['args'][1])
                    out['alu_result'] = v
                    out['write_reg'] = instr['args'][0]
                elif op == 'LOAD':
                    addr = getval(instr['args'][1])
                    out['mem_read'] = True
                    out['mem_addr'] = addr
                    out['write_reg'] = instr['args'][0]
                elif op == 'STORE':
                    addr = getval(instr['args'][1])
                    val = getval(instr['args'][0])
                    out['mem_write'] = True
                    out['mem_addr'] = addr
                    out['alu_result'] = val
                elif op == 'BEQ':
                    v1 = getval(instr['args'][0])
                    v2 = getval(instr['args'][1])
                    if v1 == v2:
                        # branch taken: update PC (note: simple model — flush IF/ID)
                        target = int(instr['args'][2])
                        self.PC = target
                        print(f"EX: BEQ taken to {target}")
                        # squash IF/ID
                        self.IF_ID = None
                        self.ID_EX = None
                        self.EX_MEM = None
                        new_EX_MEM = None
                        new_MEM_WB = None
                    else:
                        print("EX: BEQ not taken")
                elif op == 'JMP':
                    target = int(instr['args'][0])
                    self.PC = target
                    print(f"EX: JMP to {target}")
                    # flush prior stages
                    self.IF_ID = None
                    self.ID_EX = None
                    self.EX_MEM = None
                    new_EX_MEM = None
                    new_MEM_WB = None
                elif op == 'HALT':
                    out['write_reg'] = None
                    out['op'] = 'HALT'
                else:
                    pass
                if new_EX_MEM is None:
                    new_EX_MEM = out
        # -------------------- ID stage --------------------
        # Hazard detection leads to inserting bubble: convert ID_EX to None and keep IF_ID same
        insert_stall = False
        if self.IF_ID:
            id_instr = self.IF_ID.get('instr')
            if id_instr and self.detect_load_use_hazard(id_instr):
                insert_stall = True
                self.stalls += 1
                print("ID: load-use hazard detected — inserting stall (bubble)")
        if insert_stall:
            # Insert bubble: ID_EX becomes None, IF_ID remains (we will re-decode next cycle)
            new_ID_EX = None
        else:
            # normal decode
            if self.IF_ID:
                decoded = self.IF_ID['instr']
                new_ID_EX = {'instr': decoded}
            else:
                new_ID_EX = None

        # -------------------- IF stage --------------------
        new_IF_ID = None
        # Fetch next instruction into IF/ID unless a stall freezes it
        if not insert_stall:
            if 0 <= self.PC < len(self.instr_mem):
                fetched = self.instr_mem[self.PC]
                print(f"IF: Fetched PC={self.PC} -> {fetched['text']}")
                new_IF_ID = {'instr': fetched}
                self.PC += 1
            else:
                # No instruction: set fetched to HALT pseudo-instr
                new_IF_ID = None

        # commit stage updates (advance pipeline)
        self.MEM_WB = new_MEM_WB
        self.EX_MEM = new_EX_MEM
        self.ID_EX = new_ID_EX
        self.IF_ID = new_IF_ID

        print(f"Registers: {self.regs}")
        print('-'*60)

    def run(self, max_cycles=2000):
        while not self.halted and self.cycle < max_cycles:
            self.step()
        # after run write memory dump
        with open('mem_dump.txt','w') as f:
            for i,v in enumerate(self.mem):
                f.write(f"{i}: {v}\n")
        print('Memory dumped to mem_dump.txt')
        # produce plot
        if MATPLOTLIB_AVAILABLE:
            max_cycles = len(self.pc_history)
            for r in self.reg_history:
                reg_vals = self.reg_history[r]
                if len(reg_vals) < max_cycles:
                    reg_vals.extend([reg_vals[-1] if reg_vals else 0] * (max_cycles - len(reg_vals)))
            cycles = list(range(1, max_cycles+1))

            
            plt.figure(figsize=(10,6))
            for r in sorted(self.reg_history.keys()):
                plt.plot(cycles, self.reg_history[r], label=r)
            plt.xlabel('Cycle (after WB)')
            plt.ylabel('Register value')
            plt.title('Register values over cycles (R0..R7)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('registers.png')
            print('Register plot saved to registers.png')
            plt.show()
        else:
            print('matplotlib not available — skipping plot')

# ---------------------- Sample program and runner ----------------------

SAMPLE_PROGRAM = '''
; sample: compute 6*7 using repeated addition, store at mem[0]
MOV R0, #6       ; counter
MOV R1, #7       ; multiplicand
MOV R2, #0       ; result
loop:
BEQ R0, R3, end  ; branch if R0 == 0 (R3 is 0)
ADD R2, R2, R1
SUB R0, R0, #1
JMP loop
end:
STORE R2, #0
HALT
.DATA 10, #123   ; example data at mem[10]
'''

def main():
    asm = SAMPLE_PROGRAM
    if len(sys.argv) > 1:
        # treat argv[1] as filename
        with open(sys.argv[1]) as f:
            asm = f.read()
    assembler = Assembler()
    instrs, data = assembler.assemble(asm)
    print('Assembled', len(instrs), 'instructions,', len(data), 'data words')
    cpu = PipelineCPU(instrs, data, mem_size=256)
    cpu.run()
    print('Final registers:', cpu.regs)
    print('Stalls inserted:', cpu.stalls)
    print('PC history (first 20):', cpu.pc_history[:20])

if __name__ == '__main__':
    main()
