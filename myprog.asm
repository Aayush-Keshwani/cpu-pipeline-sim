.DATA
A: 5
B: 7
C: 10
D: 0
E: 1
F: 0
SUM: 0
COUNT: 5

# Program Start
# R1 = A, R2 = B, R3 = C
LOAD R1, A
LOAD R2, B
LOAD R3, C

# SUM = A + B + C
ADD R4, R1, R2
ADD R4, R4, R3
STORE R4, SUM

# D = (A * B) - C
MUL R5, R1, R2
SUB R5, R5, R3
STORE R5, D

# E = (C / B)
DIV R6, R3, R2
STORE R6, E

# Initialize F = 0
LOAD R7, F

# Loop: Add COUNT times A into F
LOAD R8, COUNT
LOOP_START:
ADD R7, R7, R1     # F = F + A
SUB R8, R8, 1      # COUNT = COUNT - 1
BNZ R8, LOOP_START # Repeat if COUNT != 0

# Save final result into F
STORE R7, F

# End
HALT
