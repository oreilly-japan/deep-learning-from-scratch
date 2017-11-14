from or_gate_pk import OR
from nand_gate_pk import NAND
from and_gate_pk import AND

def XOR(x1, x2):
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    y = AND(s1, s2)
    if y <= 0:
        return 0
    elif y > 0:
        return 1

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
