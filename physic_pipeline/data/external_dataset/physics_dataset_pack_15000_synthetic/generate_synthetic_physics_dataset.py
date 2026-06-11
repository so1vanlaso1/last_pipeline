import json, math, random, re, hashlib
from collections import Counter

SEED = 20260514
random.seed(SEED)
TARGET = 15000
K = 9e9

existing_path = '/mnt/data/clean_physics_dataset(1).jsonl'
out_path = '/mnt/data/synthetic_physics_dataset_15000.jsonl'
manifest_path = '/mnt/data/synthetic_physics_dataset_manifest.json'
report_path = '/mnt/data/synthetic_physics_dataset_validation_report.json'

# ---------- formatting helpers ----------
def fmt_sig(x, sig=3):
    if isinstance(x, int):
        return str(x)
    if abs(x) < 1e-15:
        return '0'
    # avoid scientific notation for friendly values
    if 1e-3 <= abs(x) < 1e5:
        s = f"{x:.{sig}g}"
        # convert 1e+04 style into plain if python chose e in this range
        if 'e' in s or 'E' in s:
            s = f"{x:.{max(0, sig - int(math.floor(math.log10(abs(x)))) - 1)}f}".rstrip('0').rstrip('.')
        return s
    mant, exp = f"{x:.{sig-1}e}".split('e')
    mant = mant.rstrip('0').rstrip('.')
    exp = int(exp)
    return f"{mant} × 10^{exp}"

def fmt_calc(x, unit):
    return f"{fmt_sig(x)} {unit}" if unit else fmt_sig(x)

def choice(seq):
    return random.choice(seq)

def rnd_int(a,b):
    return random.randint(a,b)

def rnd_from(seq):
    return random.choice(seq)

def make_record(i, category, question, answer, unit, cot, premises):
    rid = f"SYNPHY{i:05d}"
    return {
        'id': rid,
        'source_record_id': rid,
        'type': 'physics',
        'category': category,
        'question': question,
        'answer': answer,
        'unit': unit,
        'cot': cot,
        'explanation': cot,
        'premises': premises,
        'verification': 'Synthetic deterministic formula template; generated with fixed seed and recomputed during validation. Constants: k = 9 × 10^9 N·m²/C², π from Python math.pi where used. Answers are rounded to 3 significant figures when needed.'
    }

# ---------- category generators ----------
def gen_ohm_current():
    R = rnd_from([2,3,4,5,6,8,10,12,15,20,24,30,40,50,60,75,100,120,150,200])
    I = rnd_from([0.05,0.1,0.2,0.25,0.5,0.75,1,1.2,1.5,2,2.5,3,4,5])
    U = R*I
    q = f"A resistor has resistance R = {fmt_sig(R)} Ω and is connected to a voltage source of U = {fmt_sig(U)} V. Calculate the current through the resistor. Round to 3 significant figures if needed."
    ans = fmt_sig(I)
    cot = f"Step 1: Use Ohm's law I = U / R.\nStep 2: Substitute U = {fmt_sig(U)} V and R = {fmt_sig(R)} Ω.\nStep 3: I = {fmt_sig(U)} / {fmt_sig(R)} = {fmt_calc(I, 'A')}."
    return 'ohm_law_current', q, ans, 'A', cot, ["Ohm's law: U = IR"]

def gen_ohm_voltage():
    R = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    I = rnd_from([0.1,0.2,0.3,0.4,0.5,0.75,1,1.5,2,2.4,3,4])
    U = I*R
    q = f"A current of I = {fmt_sig(I)} A flows through a resistor of R = {fmt_sig(R)} Ω. Calculate the voltage across the resistor. Round to 3 significant figures if needed."
    ans = fmt_sig(U)
    cot = f"Step 1: Use Ohm's law U = IR.\nStep 2: Substitute I = {fmt_sig(I)} A and R = {fmt_sig(R)} Ω.\nStep 3: U = {fmt_sig(I)} × {fmt_sig(R)} = {fmt_calc(U, 'V')}."
    return 'ohm_law_voltage', q, ans, 'V', cot, ["Ohm's law: U = IR"]

def gen_ohm_resistance():
    R = rnd_from([1.5,2,3,4,5,6,8,10,12,15,20,25,30,40,50,60,80,100,150,200])
    I = rnd_from([0.05,0.1,0.2,0.25,0.4,0.5,0.75,1,1.2,1.5,2,2.5,3])
    U = R*I
    q = f"A component has voltage U = {fmt_sig(U)} V across it while the current is I = {fmt_sig(I)} A. Find its resistance. Round to 3 significant figures if needed."
    ans = fmt_sig(R)
    cot = f"Step 1: Rearrange Ohm's law to R = U / I.\nStep 2: Substitute U = {fmt_sig(U)} V and I = {fmt_sig(I)} A.\nStep 3: R = {fmt_sig(U)} / {fmt_sig(I)} = {fmt_calc(R, 'Ω')}."
    return 'ohm_law_resistance', q, ans, 'Ω', cot, ["Ohm's law: U = IR"]

def gen_power_ui():
    U = rnd_from([3,5,6,9,12,15,18,24,30,36,48,60,72,110,120,220])
    I = rnd_from([0.05,0.1,0.2,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5])
    P = U*I
    q = f"An electric device operates at voltage U = {fmt_sig(U)} V and current I = {fmt_sig(I)} A. Calculate its electric power. Round to 3 significant figures if needed."
    ans = fmt_sig(P)
    cot = f"Step 1: Use P = UI.\nStep 2: Substitute U = {fmt_sig(U)} V and I = {fmt_sig(I)} A.\nStep 3: P = {fmt_sig(U)} × {fmt_sig(I)} = {fmt_calc(P, 'W')}."
    return 'electric_power_ui', q, ans, 'W', cot, ["Electric power: P = UI"]

def gen_power_i2r():
    I = rnd_from([0.1,0.2,0.25,0.3,0.5,0.75,1,1.2,1.5,2,2.5,3,4])
    R = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    P = I*I*R
    q = f"A current I = {fmt_sig(I)} A passes through a resistor R = {fmt_sig(R)} Ω. Calculate the power dissipated in the resistor. Round to 3 significant figures if needed."
    ans = fmt_sig(P)
    cot = f"Step 1: Use the Joule power formula P = I²R.\nStep 2: Substitute I = {fmt_sig(I)} A and R = {fmt_sig(R)} Ω.\nStep 3: P = ({fmt_sig(I)})² × {fmt_sig(R)} = {fmt_calc(P, 'W')}."
    return 'resistor_power_i2r', q, ans, 'W', cot, ["Power in a resistor: P = I²R"]

def gen_power_u2r():
    R = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100,150,200])
    U = rnd_from([3,5,6,9,12,15,18,24,30,36,48,60,72,120,220])
    P = U*U/R
    q = f"A resistor R = {fmt_sig(R)} Ω is connected across U = {fmt_sig(U)} V. Calculate the power dissipated. Round to 3 significant figures if needed."
    ans = fmt_sig(P)
    cot = f"Step 1: Use P = U² / R.\nStep 2: Substitute U = {fmt_sig(U)} V and R = {fmt_sig(R)} Ω.\nStep 3: P = ({fmt_sig(U)})² / {fmt_sig(R)} = {fmt_calc(P, 'W')}."
    return 'resistor_power_u2r', q, ans, 'W', cot, ["Power in a resistor: P = U²/R"]

def gen_energy_power_time():
    P = rnd_from([2,5,10,12,15,20,25,30,40,50,60,75,100,150,200,500])
    t = rnd_from([5,10,15,20,30,45,60,90,120,180,300,600])
    E = P*t
    q = f"A device of power P = {fmt_sig(P)} W operates for t = {fmt_sig(t)} s. Calculate the electrical energy consumed. Round to 3 significant figures if needed."
    ans = fmt_sig(E)
    cot = f"Step 1: Use E = Pt.\nStep 2: Substitute P = {fmt_sig(P)} W and t = {fmt_sig(t)} s.\nStep 3: E = {fmt_sig(P)} × {fmt_sig(t)} = {fmt_calc(E, 'J')}."
    return 'electric_energy_power_time', q, ans, 'J', cot, ["Electrical energy: E = Pt"]

def gen_series_resistance():
    Rs = [rnd_from([1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100]) for _ in range(rnd_from([2,3,4]))]
    Req = sum(Rs)
    names = ', '.join([f"R{i+1} = {fmt_sig(r)} Ω" for i,r in enumerate(Rs)])
    q = f"Resistors are connected in series with {names}. Calculate the equivalent resistance. Round to 3 significant figures if needed."
    ans = fmt_sig(Req)
    cot = f"Step 1: For series resistors, equivalent resistance is the sum of all resistances.\nStep 2: R_eq = {' + '.join(fmt_sig(r) for r in Rs)}.\nStep 3: R_eq = {fmt_calc(Req, 'Ω')}."
    return 'series_resistance', q, ans, 'Ω', cot, ["Series resistors: R_eq = R1 + R2 + ..."]

def gen_parallel_two_resistance():
    R1 = rnd_from([2,3,4,5,6,8,10,12,15,20,24,30,40,50,60,75,100,120,150,200])
    R2 = rnd_from([2,3,4,5,6,8,10,12,15,20,24,30,40,50,60,75,100,120,150,200])
    Req = R1*R2/(R1+R2)
    q = f"Two resistors R1 = {fmt_sig(R1)} Ω and R2 = {fmt_sig(R2)} Ω are connected in parallel. Calculate the equivalent resistance. Round to 3 significant figures if needed."
    ans = fmt_sig(Req)
    cot = f"Step 1: For two parallel resistors, R_eq = R1R2 / (R1 + R2).\nStep 2: Substitute R1 = {fmt_sig(R1)} Ω and R2 = {fmt_sig(R2)} Ω.\nStep 3: R_eq = {fmt_sig(R1)} × {fmt_sig(R2)} / ({fmt_sig(R1)} + {fmt_sig(R2)}) = {fmt_calc(Req, 'Ω')}."
    return 'parallel_two_resistance', q, ans, 'Ω', cot, ["Two parallel resistors: R_eq = R1R2/(R1+R2)"]

def gen_parallel_three_resistance():
    Rs = [rnd_from([3,4,5,6,8,10,12,15,20,24,30,40,50,60,75,100,120,150]) for _ in range(3)]
    inv = sum(1/r for r in Rs)
    Req = 1/inv
    q = f"Three resistors R1 = {fmt_sig(Rs[0])} Ω, R2 = {fmt_sig(Rs[1])} Ω, and R3 = {fmt_sig(Rs[2])} Ω are connected in parallel. Calculate the equivalent resistance. Round to 3 significant figures if needed."
    ans = fmt_sig(Req)
    cot = f"Step 1: For parallel resistors, 1/R_eq = 1/R1 + 1/R2 + 1/R3.\nStep 2: 1/R_eq = 1/{fmt_sig(Rs[0])} + 1/{fmt_sig(Rs[1])} + 1/{fmt_sig(Rs[2])}.\nStep 3: R_eq = {fmt_calc(Req, 'Ω')}."
    return 'parallel_three_resistance', q, ans, 'Ω', cot, ["Parallel resistors: 1/R_eq = Σ(1/R_i)"]

def gen_series_circuit_current():
    R1 = rnd_from([2,3,4,5,6,8,10,12,15,20,24,30,40,50])
    R2 = rnd_from([2,3,4,5,6,8,10,12,15,20,24,30,40,50])
    R3 = rnd_from([0,0,3,5,10,15,20])
    U = rnd_from([6,9,12,15,18,24,30,36,48,60,72,120])
    Req = R1+R2+R3
    I = U/Req
    parts = f"R1 = {fmt_sig(R1)} Ω, R2 = {fmt_sig(R2)} Ω" + (f", R3 = {fmt_sig(R3)} Ω" if R3 else '')
    q = f"A source of U = {fmt_sig(U)} V is connected to series resistors {parts}. Calculate the circuit current. Round to 3 significant figures if needed."
    ans = fmt_sig(I)
    cot = f"Step 1: For series resistors, R_eq = {fmt_sig(Req)} Ω.\nStep 2: Use Ohm's law I = U / R_eq.\nStep 3: I = {fmt_sig(U)} / {fmt_sig(Req)} = {fmt_calc(I, 'A')}."
    return 'series_circuit_current', q, ans, 'A', cot, ["Series resistors add", "Ohm's law: I = U/R"]

def gen_voltage_divider():
    R1 = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    R2 = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    U = rnd_from([5,6,9,12,15,18,24,30,36,48,60])
    U2 = U*R2/(R1+R2)
    q = f"Two resistors R1 = {fmt_sig(R1)} Ω and R2 = {fmt_sig(R2)} Ω are connected in series across U = {fmt_sig(U)} V. Calculate the voltage across R2. Round to 3 significant figures if needed."
    ans = fmt_sig(U2)
    cot = f"Step 1: In a series voltage divider, U2 = U × R2 / (R1 + R2).\nStep 2: Substitute U = {fmt_sig(U)} V, R1 = {fmt_sig(R1)} Ω, and R2 = {fmt_sig(R2)} Ω.\nStep 3: U2 = {fmt_sig(U)} × {fmt_sig(R2)} / ({fmt_sig(R1)} + {fmt_sig(R2)}) = {fmt_calc(U2, 'V')}."
    return 'voltage_divider', q, ans, 'V', cot, ["Voltage divider: U_i = U_total R_i/R_total"]

def gen_parallel_branch_current():
    U = rnd_from([3,5,6,9,12,15,18,24,30,36,48,60,72])
    R = rnd_from([2,3,4,5,6,8,10,12,15,20,24,30,40,50,60,75,100,150])
    I = U/R
    q = f"A resistor branch of R = {fmt_sig(R)} Ω is connected in parallel across a source of U = {fmt_sig(U)} V. Calculate the current in that branch. Round to 3 significant figures if needed."
    ans = fmt_sig(I)
    cot = f"Step 1: In a parallel branch, the branch voltage equals the source voltage.\nStep 2: Use I = U / R.\nStep 3: I = {fmt_sig(U)} / {fmt_sig(R)} = {fmt_calc(I, 'A')}."
    return 'parallel_branch_current', q, ans, 'A', cot, ["Parallel branches have the same voltage", "Ohm's law: I = U/R"]

def gen_current_divider_two():
    It = rnd_from([0.5,0.75,1,1.2,1.5,2,2.5,3,4,5,6,8,10])
    R1 = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    R2 = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    I1 = It*R2/(R1+R2)
    q = f"A total current I = {fmt_sig(It)} A splits between two parallel resistors R1 = {fmt_sig(R1)} Ω and R2 = {fmt_sig(R2)} Ω. Calculate the current through R1. Round to 3 significant figures if needed."
    ans = fmt_sig(I1)
    cot = f"Step 1: For two parallel resistors, current through R1 is I1 = I × R2 / (R1 + R2).\nStep 2: Substitute I = {fmt_sig(It)} A, R1 = {fmt_sig(R1)} Ω, and R2 = {fmt_sig(R2)} Ω.\nStep 3: I1 = {fmt_sig(It)} × {fmt_sig(R2)} / ({fmt_sig(R1)} + {fmt_sig(R2)}) = {fmt_calc(I1, 'A')}."
    return 'current_divider_two', q, ans, 'A', cot, ["Current divider: I1 = I_total R2/(R1+R2)"]

def gen_capacitor_charge():
    C = rnd_from([0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470,1000])
    U = rnd_from([1.5,3,5,6,9,12,15,18,24,30,36,48,60,100])
    Q = C*U # microcoulomb
    q = f"A capacitor has capacitance C = {fmt_sig(C)} μF and voltage U = {fmt_sig(U)} V. Calculate the charge stored on the capacitor. Round to 3 significant figures if needed."
    ans = fmt_sig(Q)
    cot = f"Step 1: Use Q = CU.\nStep 2: With C in μF and U in V, Q is in μC.\nStep 3: Q = {fmt_sig(C)} × {fmt_sig(U)} = {fmt_calc(Q, 'μC')}."
    return 'capacitor_charge', q, ans, 'μC', cot, ["Capacitor charge: Q = CU"]

def gen_capacitor_voltage_from_charge():
    C = rnd_from([0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470])
    U = rnd_from([1.5,3,5,6,9,12,15,18,24,30,36,48,60])
    Q = C*U
    q = f"A capacitor stores charge Q = {fmt_sig(Q)} μC and has capacitance C = {fmt_sig(C)} μF. Calculate the voltage across it. Round to 3 significant figures if needed."
    ans = fmt_sig(U)
    cot = f"Step 1: Rearrange Q = CU to U = Q / C.\nStep 2: Substitute Q = {fmt_sig(Q)} μC and C = {fmt_sig(C)} μF.\nStep 3: U = {fmt_sig(Q)} / {fmt_sig(C)} = {fmt_calc(U, 'V')}."
    return 'capacitor_voltage_from_charge', q, ans, 'V', cot, ["Capacitor charge: Q = CU"]

def gen_capacitor_energy():
    C = rnd_from([0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470,1000])
    U = rnd_from([1.5,3,5,6,9,12,15,18,24,30,36,48,60,100])
    E = 0.5*C*U*U # μJ
    q = f"Calculate the energy stored in a capacitor when C = {fmt_sig(C)} μF and U = {fmt_sig(U)} V. Round to 3 significant figures if needed."
    ans = fmt_sig(E)
    cot = f"Step 1: Use the capacitor energy formula E = 0.5CU².\nStep 2: With C in μF and U in V, E is in μJ.\nStep 3: E = 0.5 × {fmt_sig(C)} × ({fmt_sig(U)})² = {fmt_calc(E, 'μJ')}."
    return 'capacitor_energy', q, ans, 'μJ', cot, ["Capacitor energy: E = 0.5CU²"]

def gen_capacitance_parallel():
    Cs = [rnd_from([0.47,1,2.2,4.7,10,22,47,100,220,470]) for _ in range(rnd_from([2,3,4]))]
    Ceq = sum(Cs)
    names = ', '.join([f"C{i+1} = {fmt_sig(c)} μF" for i,c in enumerate(Cs)])
    q = f"Capacitors are connected in parallel with {names}. Calculate the equivalent capacitance. Round to 3 significant figures if needed."
    ans = fmt_sig(Ceq)
    cot = f"Step 1: For parallel capacitors, capacitances add directly.\nStep 2: C_eq = {' + '.join(fmt_sig(c) for c in Cs)}.\nStep 3: C_eq = {fmt_calc(Ceq, 'μF')}."
    return 'parallel_capacitance', q, ans, 'μF', cot, ["Parallel capacitors: C_eq = C1 + C2 + ..."]

def gen_capacitance_series_two():
    C1 = rnd_from([0.47,1,2.2,4.7,10,22,47,100,220,470])
    C2 = rnd_from([0.47,1,2.2,4.7,10,22,47,100,220,470])
    Ceq = C1*C2/(C1+C2)
    q = f"Two capacitors C1 = {fmt_sig(C1)} μF and C2 = {fmt_sig(C2)} μF are connected in series. Calculate the equivalent capacitance. Round to 3 significant figures if needed."
    ans = fmt_sig(Ceq)
    cot = f"Step 1: For two series capacitors, C_eq = C1C2 / (C1 + C2).\nStep 2: Substitute C1 = {fmt_sig(C1)} μF and C2 = {fmt_sig(C2)} μF.\nStep 3: C_eq = {fmt_sig(C1)} × {fmt_sig(C2)} / ({fmt_sig(C1)} + {fmt_sig(C2)}) = {fmt_calc(Ceq, 'μF')}."
    return 'series_two_capacitance', q, ans, 'μF', cot, ["Two series capacitors: C_eq = C1C2/(C1+C2)"]

def gen_capacitor_sharing_equal():
    C = rnd_from([1,2,2.2,4.7,10,22,47,100,220])
    U = rnd_from([3,5,6,9,12,15,18,24,30,36,48])
    n = rnd_from([2,3,4,5])
    Q = C*U
    Ctotal = n*C
    Efinal = Q*Q/(2*Ctotal)  # μJ, since μC^2/μF = μJ
    q = f"A {fmt_sig(C)} μF capacitor is charged to {fmt_sig(U)} V, disconnected, and then connected in parallel with {n-1} identical uncharged capacitor(s). Calculate the total final stored energy. Round to 3 significant figures if needed."
    ans = fmt_sig(Efinal)
    cot = f"Step 1: Initial charge is Q = CU = {fmt_sig(C)} × {fmt_sig(U)} = {fmt_sig(Q)} μC.\nStep 2: After sharing among {n} identical capacitors in parallel, C_total = {n}C = {fmt_sig(Ctotal)} μF.\nStep 3: Final energy is E = Q² / (2C_total).\nStep 4: E = ({fmt_sig(Q)})² / (2 × {fmt_sig(Ctotal)}) = {fmt_calc(Efinal, 'μJ')}."
    return 'capacitor_charge_sharing_energy', q, ans, 'μJ', cot, ["Charge conservation", "Final energy: E = Q²/(2C_total)"]

def gen_point_charge_field():
    qn = rnd_from([1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100,150,200]) # nC
    rcm = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    er = rnd_from([1,1,1,1,2,2.2,3,4,5])
    E = K/er*(qn*1e-9)/( (rcm/100)**2 )
    medium = 'vacuum' if er == 1 else f"a medium with dielectric constant εr = {fmt_sig(er)}"
    qtxt = f"A point charge q = {fmt_sig(qn)} nC is in {medium}. Calculate the magnitude of the electric field at distance r = {fmt_sig(rcm)} cm. Use k = 9 × 10^9 N·m²/C². Round to 3 significant figures if needed."
    ans = fmt_sig(E)
    cot = f"Step 1: Use E = (k/εr)|q|/r².\nStep 2: Convert q = {fmt_sig(qn)} nC = {fmt_sig(qn*1e-9)} C and r = {fmt_sig(rcm)} cm = {fmt_sig(rcm/100)} m.\nStep 3: E = (9 × 10^9 / {fmt_sig(er)}) × {fmt_sig(qn*1e-9)} / ({fmt_sig(rcm/100)})² = {fmt_calc(E, 'N/C')}."
    return 'point_charge_electric_field', qtxt, ans, 'N/C', cot, ["Point charge field: E = k|q|/(εr r²)"]

def gen_coulomb_force():
    q1u = rnd_from([0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,4,5,6,8,10])
    q2u = rnd_from([0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,4,5,6,8,10])
    rcm = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50])
    er = rnd_from([1,1,1,2,2.5,3,4,5])
    F = K/er*(q1u*1e-6)*(q2u*1e-6)/( (rcm/100)**2 )
    qtxt = f"Two point charges have magnitudes q1 = {fmt_sig(q1u)} μC and q2 = {fmt_sig(q2u)} μC, separated by r = {fmt_sig(rcm)} cm in a medium with εr = {fmt_sig(er)}. Calculate the magnitude of the electrostatic force. Use k = 9 × 10^9 N·m²/C². Round to 3 significant figures if needed."
    ans = fmt_sig(F)
    cot = f"Step 1: Use Coulomb's law F = (k/εr)|q1q2|/r².\nStep 2: Convert charges to coulombs and distance to meters.\nStep 3: F = (9 × 10^9 / {fmt_sig(er)}) × ({fmt_sig(q1u*1e-6)}) × ({fmt_sig(q2u*1e-6)}) / ({fmt_sig(rcm/100)})² = {fmt_calc(F, 'N')}."
    return 'coulomb_force', qtxt, ans, 'N', cot, ["Coulomb's law: F = k|q1q2|/(εr r²)"]

def gen_electric_potential_point_charge():
    qn = rnd_from([-200,-150,-100,-75,-50,-30,-20,-10,-5,5,10,20,30,50,75,100,150,200])
    rcm = rnd_from([2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    er = rnd_from([1,1,1,2,2.2,3,4,5])
    V = K/er*(qn*1e-9)/(rcm/100)
    qtxt = f"A point charge q = {fmt_sig(qn)} nC is placed in a medium with εr = {fmt_sig(er)}. Find the electric potential at a point {fmt_sig(rcm)} cm from the charge. Use k = 9 × 10^9 N·m²/C². Round to 3 significant figures if needed."
    ans = fmt_sig(V)
    cot = f"Step 1: Use V = (k/εr)q/r, keeping the sign of q.\nStep 2: Convert q = {fmt_sig(qn)} nC = {fmt_sig(qn*1e-9)} C and r = {fmt_sig(rcm/100)} m.\nStep 3: V = (9 × 10^9 / {fmt_sig(er)}) × {fmt_sig(qn*1e-9)} / {fmt_sig(rcm/100)} = {fmt_calc(V, 'V')}."
    return 'point_charge_potential', qtxt, ans, 'V', cot, ["Point charge potential: V = kq/(εr r)"]

def gen_electric_work_qu():
    qmicro = rnd_from([-20,-10,-5,-2,-1,1,2,5,10,20,50,100,200,500])
    U = rnd_from([3,5,6,9,12,15,18,24,30,36,48,60,100,220])
    W = qmicro*U # μJ signed
    qtxt = f"A charge q = {fmt_sig(qmicro)} μC moves through a potential difference U = {fmt_sig(U)} V. Calculate the work W = qU. Round to 3 significant figures if needed."
    ans = fmt_sig(W)
    cot = f"Step 1: Use W = qU.\nStep 2: With q in μC and U in V, W is in μJ.\nStep 3: W = {fmt_sig(qmicro)} × {fmt_sig(U)} = {fmt_calc(W, 'μJ')}."
    return 'electric_work_charge_voltage', qtxt, ans, 'μJ', cot, ["Work across potential difference: W = qU"]

def gen_uniform_field_plate():
    U = rnd_from([3,5,6,9,12,15,18,24,30,50,100,150,200,300,500,1000])
    dcm = rnd_from([0.1,0.2,0.5,1,1.5,2,3,4,5,8,10,12,15,20])
    E = U/(dcm/100)
    qtxt = f"Two parallel plates have potential difference U = {fmt_sig(U)} V and separation d = {fmt_sig(dcm)} cm. Calculate the uniform electric field between the plates. Round to 3 significant figures if needed."
    ans = fmt_sig(E)
    cot = f"Step 1: Use E = U/d for a uniform field.\nStep 2: Convert d = {fmt_sig(dcm)} cm = {fmt_sig(dcm/100)} m.\nStep 3: E = {fmt_sig(U)} / {fmt_sig(dcm/100)} = {fmt_calc(E, 'V/m')}."
    return 'parallel_plate_uniform_field', qtxt, ans, 'V/m', cot, ["Uniform field between plates: E = U/d"]

def gen_force_in_uniform_field():
    qmicro = rnd_from([0.1,0.2,0.5,1,2,3,5,10,20,50,100])
    E = rnd_from([100,200,500,1000,1500,2000,5000,10000,20000,50000])
    FuN = qmicro*E # μN
    qtxt = f"A charge q = {fmt_sig(qmicro)} μC is placed in a uniform electric field E = {fmt_sig(E)} N/C. Calculate the magnitude of the electric force. Round to 3 significant figures if needed."
    ans = fmt_sig(FuN)
    cot = f"Step 1: Use F = qE.\nStep 2: With q in μC and E in N/C, F is in μN.\nStep 3: F = {fmt_sig(qmicro)} × {fmt_sig(E)} = {fmt_calc(FuN, 'μN')}."
    return 'force_in_uniform_electric_field', qtxt, ans, 'μN', cot, ["Electric force: F = qE"]

def gen_inductor_energy():
    LmH = rnd_from([0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470])
    I = rnd_from([0.1,0.2,0.3,0.5,0.75,1,1.5,2,2.5,3,4,5,8,10])
    EmJ = 0.5*LmH*I*I # mJ because mH*A^2
    q = f"An inductor has inductance L = {fmt_sig(LmH)} mH and carries current I = {fmt_sig(I)} A. Calculate the magnetic energy stored. Round to 3 significant figures if needed."
    ans = fmt_sig(EmJ)
    cot = f"Step 1: Use W = 0.5LI².\nStep 2: With L in mH and I in A, W is in mJ.\nStep 3: W = 0.5 × {fmt_sig(LmH)} × ({fmt_sig(I)})² = {fmt_calc(EmJ, 'mJ')}."
    return 'inductor_energy', q, ans, 'mJ', cot, ["Inductor energy: W = 0.5LI²"]

def gen_rl_time_constant():
    LmH = rnd_from([1,2.2,4.7,10,22,47,100,220,470,1000])
    R = rnd_from([1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100])
    tau_ms = LmH/R # because mH/ohm = ms
    q = f"An RL circuit has inductance L = {fmt_sig(LmH)} mH and resistance R = {fmt_sig(R)} Ω. Calculate the time constant τ. Round to 3 significant figures if needed."
    ans = fmt_sig(tau_ms)
    cot = f"Step 1: For an RL circuit, τ = L/R.\nStep 2: With L in mH and R in Ω, τ is in ms.\nStep 3: τ = {fmt_sig(LmH)} / {fmt_sig(R)} = {fmt_calc(tau_ms, 'ms')}."
    return 'rl_time_constant', q, ans, 'ms', cot, ["RL time constant: τ = L/R"]

def gen_rc_time_constant():
    Rk = rnd_from([0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470])
    Cu = rnd_from([0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470])
    tau_ms = Rk*Cu # kΩ*μF = ms
    q = f"An RC circuit has resistance R = {fmt_sig(Rk)} kΩ and capacitance C = {fmt_sig(Cu)} μF. Calculate the time constant τ. Round to 3 significant figures if needed."
    ans = fmt_sig(tau_ms)
    cot = f"Step 1: For an RC circuit, τ = RC.\nStep 2: With R in kΩ and C in μF, τ is in ms.\nStep 3: τ = {fmt_sig(Rk)} × {fmt_sig(Cu)} = {fmt_calc(tau_ms, 'ms')}."
    return 'rc_time_constant', q, ans, 'ms', cot, ["RC time constant: τ = RC"]

def gen_capacitive_reactance():
    f = rnd_from([50,60,100,120,400,500,1000,2000,5000,10000,20000])
    Cu = rnd_from([0.01,0.022,0.047,0.1,0.22,0.47,1,2.2,4.7,10,22,47,100])
    Xc = 1/(2*math.pi*f*Cu*1e-6)
    q = f"A capacitor C = {fmt_sig(Cu)} μF is connected to an AC signal of frequency f = {fmt_sig(f)} Hz. Calculate the capacitive reactance Xc. Round to 3 significant figures if needed."
    ans = fmt_sig(Xc)
    cot = f"Step 1: Use Xc = 1 / (2πfC).\nStep 2: Convert C = {fmt_sig(Cu)} μF = {fmt_sig(Cu*1e-6)} F.\nStep 3: Xc = 1 / (2π × {fmt_sig(f)} × {fmt_sig(Cu*1e-6)}) = {fmt_calc(Xc, 'Ω')}."
    return 'capacitive_reactance', q, ans, 'Ω', cot, ["Capacitive reactance: Xc = 1/(2πfC)"]

def gen_inductive_reactance():
    f = rnd_from([50,60,100,120,400,500,1000,2000,5000,10000,20000])
    Lm = rnd_from([0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470])
    XL = 2*math.pi*f*Lm*1e-3
    q = f"An inductor L = {fmt_sig(Lm)} mH is connected to an AC signal of frequency f = {fmt_sig(f)} Hz. Calculate the inductive reactance XL. Round to 3 significant figures if needed."
    ans = fmt_sig(XL)
    cot = f"Step 1: Use XL = 2πfL.\nStep 2: Convert L = {fmt_sig(Lm)} mH = {fmt_sig(Lm*1e-3)} H.\nStep 3: XL = 2π × {fmt_sig(f)} × {fmt_sig(Lm*1e-3)} = {fmt_calc(XL, 'Ω')}."
    return 'inductive_reactance', q, ans, 'Ω', cot, ["Inductive reactance: XL = 2πfL"]

def gen_resistor_temperature():
    R0 = rnd_from([10,20,50,100,120,150,200,220,330,470,1000])
    alpha = rnd_from([0.0039,0.004,0.0043,0.005])
    dT = rnd_from([5,10,15,20,25,30,40,50,60,75,100])
    R = R0*(1+alpha*dT)
    q = f"A resistor has R0 = {fmt_sig(R0)} Ω at the reference temperature. Its temperature coefficient is α = {fmt_sig(alpha)} °C^-1 and the temperature increases by ΔT = {fmt_sig(dT)} °C. Calculate the new resistance. Round to 3 significant figures if needed."
    ans = fmt_sig(R)
    cot = f"Step 1: Use R = R0(1 + αΔT).\nStep 2: Substitute R0 = {fmt_sig(R0)} Ω, α = {fmt_sig(alpha)} °C^-1, and ΔT = {fmt_sig(dT)} °C.\nStep 3: R = {fmt_sig(R0)} × (1 + {fmt_sig(alpha)} × {fmt_sig(dT)}) = {fmt_calc(R, 'Ω')}."
    return 'resistance_temperature_dependence', q, ans, 'Ω', cot, ["Temperature dependence: R = R0(1 + αΔT)"]

def gen_battery_terminal_voltage():
    # Choose physically valid discharge cases with positive terminal voltage.
    while True:
        E = rnd_from([1.5,3,4.5,6,9,12,15,18,24,36,48])
        r = rnd_from([0.05,0.1,0.2,0.25,0.5,0.75,1,1.5,2])
        I = rnd_from([0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,4,5])
        U = E-I*r
        if U > 0:
            break
    q = f"A battery has emf E = {fmt_sig(E)} V and internal resistance r = {fmt_sig(r)} Ω. It delivers current I = {fmt_sig(I)} A. Calculate the terminal voltage. Round to 3 significant figures if needed."
    ans = fmt_sig(U)
    cot = f"Step 1: For a discharging battery, terminal voltage is U = E - Ir.\nStep 2: Substitute E = {fmt_sig(E)} V, I = {fmt_sig(I)} A, and r = {fmt_sig(r)} Ω.\nStep 3: U = {fmt_sig(E)} - {fmt_sig(I)} × {fmt_sig(r)} = {fmt_calc(U, 'V')}."
    return 'battery_terminal_voltage', q, ans, 'V', cot, ["Terminal voltage: U = E - Ir"]

def gen_internal_resistance():
    r = rnd_from([0.05,0.1,0.2,0.25,0.5,0.75,1,1.5,2,3])
    I = rnd_from([0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,4,5])
    E = rnd_from([3,4.5,6,9,12,15,18,24,36,48])
    U = E-I*r
    if U <= 0:
        E = 48
        U = E-I*r
    q = f"A battery has emf E = {fmt_sig(E)} V and terminal voltage U = {fmt_sig(U)} V while delivering current I = {fmt_sig(I)} A. Find its internal resistance. Round to 3 significant figures if needed."
    ans = fmt_sig(r)
    cot = f"Step 1: Use U = E - Ir, so r = (E - U) / I.\nStep 2: Substitute E = {fmt_sig(E)} V, U = {fmt_sig(U)} V, and I = {fmt_sig(I)} A.\nStep 3: r = ({fmt_sig(E)} - {fmt_sig(U)}) / {fmt_sig(I)} = {fmt_calc(r, 'Ω')}."
    return 'battery_internal_resistance', q, ans, 'Ω', cot, ["Terminal voltage: U = E - Ir"]

def gen_measurement_average_mae():
    base = rnd_from([0.5,1,1.5,2,2.5,3,4,5,10,12,15,20])
    delta = rnd_from([0.05,0.1,0.2,0.25,0.5,1])
    vals = [base-delta, base, base+delta]
    avg = sum(vals)/3
    mae = sum(abs(v-avg) for v in vals)/3
    unit = rnd_from(['A','V','Ω'])
    q = f"Three {unit} measurements were taken: {fmt_sig(vals[0])} {unit}, {fmt_sig(vals[1])} {unit}, {fmt_sig(vals[2])} {unit}. Calculate the average value and the mean absolute error. Round to 3 significant figures if needed."
    ans = f"{fmt_sig(avg)}; {fmt_sig(mae)}"
    cot = f"Step 1: Calculate the average: ({fmt_sig(vals[0])} + {fmt_sig(vals[1])} + {fmt_sig(vals[2])}) / 3 = {fmt_sig(avg)} {unit}.\nStep 2: Calculate deviations from the average: {fmt_sig(abs(vals[0]-avg))}, {fmt_sig(abs(vals[1]-avg))}, and {fmt_sig(abs(vals[2]-avg))} {unit}.\nStep 3: Mean absolute error = ({fmt_sig(abs(vals[0]-avg))} + {fmt_sig(abs(vals[1]-avg))} + {fmt_sig(abs(vals[2]-avg))}) / 3 = {fmt_sig(mae)} {unit}."
    return 'measurement_average_mean_absolute_error', q, ans, f"{unit}; {unit}", cot, ["Average = sum/n", "Mean absolute error = average absolute deviation"]

def gen_equivalent_conductance_parallel():
    Gs = [rnd_from([0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]) for _ in range(rnd_from([2,3,4]))]
    Geq = sum(Gs)
    names = ', '.join([f"G{i+1} = {fmt_sig(g)} S" for i,g in enumerate(Gs)])
    q = f"Conductances are connected in parallel with {names}. Calculate the equivalent conductance. Round to 3 significant figures if needed."
    ans = fmt_sig(Geq)
    cot = f"Step 1: For parallel conductances, conductances add directly.\nStep 2: G_eq = {' + '.join(fmt_sig(g) for g in Gs)}.\nStep 3: G_eq = {fmt_calc(Geq, 'S')}."
    return 'parallel_conductance', q, ans, 'S', cot, ["Parallel conductances: G_eq = G1 + G2 + ..."]

GENERATORS = [
    gen_ohm_current, gen_ohm_voltage, gen_ohm_resistance,
    gen_power_ui, gen_power_i2r, gen_power_u2r, gen_energy_power_time,
    gen_series_resistance, gen_parallel_two_resistance, gen_parallel_three_resistance,
    gen_series_circuit_current, gen_voltage_divider, gen_parallel_branch_current, gen_current_divider_two,
    gen_capacitor_charge, gen_capacitor_voltage_from_charge, gen_capacitor_energy,
    gen_capacitance_parallel, gen_capacitance_series_two, gen_capacitor_sharing_equal,
    gen_point_charge_field, gen_coulomb_force, gen_electric_potential_point_charge,
    gen_electric_work_qu, gen_uniform_field_plate, gen_force_in_uniform_field,
    gen_inductor_energy, gen_rl_time_constant, gen_rc_time_constant,
    gen_capacitive_reactance, gen_inductive_reactance,
    gen_resistor_temperature, gen_battery_terminal_voltage, gen_internal_resistance,
    gen_measurement_average_mae, gen_equivalent_conductance_parallel,
]

# Read existing questions to make generated set non-identical by exact text.
existing_questions = set()
try:
    with open(existing_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                existing_questions.add(json.loads(line).get('question','').strip())
except FileNotFoundError:
    pass

records = []
seen_q = set(existing_questions)
attempts = 0
while len(records) < TARGET and attempts < TARGET * 200:
    attempts += 1
    gen = GENERATORS[len(records) % len(GENERATORS)] if attempts < TARGET*2 else rnd_from(GENERATORS)
    cat, q, ans, unit, cot, premises = gen()
    if q in seen_q:
        continue
    seen_q.add(q)
    records.append(make_record(len(records)+1, cat, q, ans, unit, cot, premises))

if len(records) < TARGET:
    raise RuntimeError(f"Could only generate {len(records)} unique records")

with open(out_path, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# Validation: JSONL readback, schema, duplicate IDs/questions, answer/unit/cot present, no question overlap.
readback = []
with open(out_path, encoding='utf-8') as f:
    for line_no, line in enumerate(f, 1):
        obj = json.loads(line)
        readback.append(obj)
        for key in ['id','source_record_id','type','category','question','answer','unit','cot','explanation','premises','verification']:
            assert key in obj, (line_no, key)
        assert obj['type'] == 'physics'
        assert obj['question'].strip()
        assert obj['answer'].strip()
        assert obj['cot'].startswith('Step 1:')

ids = [r['id'] for r in readback]
qs = [r['question'] for r in readback]
assert len(ids) == len(set(ids))
assert len(qs) == len(set(qs))
overlap = sum(1 for q in qs if q in existing_questions)
assert overlap == 0

cat_counts = Counter(r['category'] for r in readback)
unit_counts = Counter(r['unit'] for r in readback)
sha256 = hashlib.sha256(open(out_path,'rb').read()).hexdigest()
manifest = {
    'name': 'synthetic_physics_dataset_15000',
    'created_for': 'EXACT-style physics solver training augmentation',
    'record_count': len(readback),
    'format': ['id','source_record_id','type','category','question','answer','unit','cot','explanation','premises','verification'],
    'source_type': 'synthetic_formula_templates_not_scraped',
    'license_note': 'No copied problem text from external websites. Generated from standard physics formulas; disclose as synthetic deterministic formula-template data if used in competition documentation.',
    'seed': SEED,
    'constants': {'k': '9 × 10^9 N·m²/C²', 'pi': 'Python math.pi'},
    'rounding_policy': 'Questions request rounding to 3 significant figures if needed; answer strings follow that policy.',
    'sha256_jsonl': sha256,
}
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

report = {
    'jsonl_readback_ok': True,
    'record_count': len(readback),
    'unique_ids': len(set(ids)),
    'unique_questions': len(set(qs)),
    'exact_question_overlap_with_uploaded_dataset': overlap,
    'category_count': len(cat_counts),
    'category_distribution': dict(sorted(cat_counts.items())),
    'unit_distribution': dict(unit_counts.most_common()),
    'sha256_jsonl': sha256,
    'sample_records': readback[:3],
}
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps({
    'out_path': out_path,
    'manifest_path': manifest_path,
    'report_path': report_path,
    'records': len(readback),
    'categories': len(cat_counts),
    'overlap': overlap,
    'sha256': sha256,
    'top_categories': cat_counts.most_common(5),
}, ensure_ascii=False, indent=2))
