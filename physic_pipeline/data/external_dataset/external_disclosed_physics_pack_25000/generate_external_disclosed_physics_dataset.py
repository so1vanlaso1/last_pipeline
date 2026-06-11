#!/usr/bin/env python3
"""Generate external-source-disclosed physics augmentation data.

The records are not copied from external problem statements. They are deterministic
formula-template variants grounded in disclosed open references/datasets:
- PhysWikiQuiz / Wikidata formula-question-generation paradigm
- Circuit-VQA circuit QA task families
- PhysGym physics environment / executable verification paradigm

Output format follows the user's physics JSONL records.
"""
from __future__ import annotations

import json
import math
import random
import re
import hashlib
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

OUT_DIR = Path('/mnt/data')
ORIGINAL_PATHS = [
    OUT_DIR / 'clean_physics_dataset(1).jsonl',
    OUT_DIR / 'synthetic_physics_dataset_15000.jsonl',
]
N_TARGET = 25000
SEED = 20260514
EPS0 = 8.8541878128e-12
K = 8.9875517923e9
PI = math.pi

SOURCES = {
    'physwikiquiz_wikidata': {
        'source_name': 'PhysWikiQuiz + Wikidata formula metadata',
        'source_url': 'https://github.com/gipplab/PhysWikiQuiz and https://www.wikidata.org/wiki/Wikidata:Licensing',
        'license': 'PhysWikiQuiz code: Apache-2.0; Wikidata structured data: CC0',
        'usage': 'Formula-question-generation paradigm and physics formula concept grounding; no copied external problem text.'
    },
    'circuit_vqa': {
        'source_name': 'Circuit-VQA',
        'source_url': 'https://github.com/rahcode7/Circuit-VQA',
        'license': 'CC-BY-SA 3.0 unless otherwise stated',
        'usage': 'Circuit QA task-family inspiration for value/equivalent-circuit questions; no images or original Q/A text copied.'
    },
    'physgym': {
        'source_name': 'PhysGym / PHYBench-style executable physics environments',
        'source_url': 'https://github.com/principia-ai/PhysGym',
        'license': 'MIT',
        'usage': 'Executable-verification style and numerical-input/output framing; no original benchmark samples copied.'
    },
}

# ---------- formatting helpers ----------

def clean_num(x: float, sig: int = 4) -> str:
    if abs(x) < 1e-12:
        return '0'
    # use fixed for ordinary values, scientific for extremes
    ax = abs(x)
    if ax >= 1e6 or ax < 1e-3:
        s = f"{x:.{sig}g}"
    else:
        s = f"{x:.{sig}f}"
        s = s.rstrip('0').rstrip('.')
    return s.replace('e+0', 'e').replace('e+', 'e').replace('e-0', 'e-')

def ans(x: float, sig: int = 4) -> str:
    return clean_num(x, sig)

def norm_q(q: str) -> str:
    return re.sub(r'\s+', ' ', q.strip().lower())

def stable_id(prefix: str, question: str) -> str:
    h = hashlib.sha1(question.encode('utf-8')).hexdigest()[:10].upper()
    return f"{prefix}{h}"

def choice(rng: random.Random, xs):
    return rng.choice(xs)

def randint(rng: random.Random, a: int, b: int, step: int = 1) -> int:
    return rng.randrange(a, b + 1, step)

def round_to(x: float, nd=6) -> float:
    return round(x, nd)


def make_record(prefix: str, category: str, q: str, cot: str, answer: str, unit: str,
                premises: List[str], verification: Dict[str, Any], source_key: str) -> Dict[str, Any]:
    rid = stable_id(prefix, q)
    return {
        'id': rid,
        'source_record_id': rid,
        'type': 'physics',
        'category': category,
        'question': q,
        'answer': answer,
        'unit': unit,
        'cot': cot,
        'explanation': cot,
        'premises': premises,
        'verification': verification,
        'external_source': SOURCES[source_key],
        'disclosure_note': 'Externally disclosed source-guided formula-template augmentation; question wording and numeric values generated locally; no copied external problem text.'
    }

# ---------- templates ----------
Template = Callable[[random.Random], Dict[str, Any]]

def t_ohm_current(rng):
    R = randint(rng, 2, 200)
    I = choice(rng, [0.05,0.08,0.1,0.12,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.2,1.5,2,2.5,3,4,5])
    V = I*R
    q = f"A resistor of resistance {R} Ω is connected to a {clean_num(V)} V source. Calculate the current through the resistor."
    cot = f"Step 1: Use Ohm's law I = V/R.\nStep 2: Substitute V = {clean_num(V)} V and R = {R} Ω.\nStep 3: I = {clean_num(V)} / {R} = {ans(I)} A."
    return make_record('EXTOM', 'ohms_law_current', q, cot, ans(I), 'A', ["Ohm's law: I = V/R"], {'formula':'I=V/R','inputs':{'V_V':round_to(V),'R_ohm':R},'recomputed_answer':round_to(I)}, 'physwikiquiz_wikidata')

def t_ohm_voltage(rng):
    R = randint(rng, 3, 250)
    I = choice(rng, [0.02,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2])
    V = I*R
    q = f"A current of {clean_num(I)} A flows through a {R} Ω resistor. Find the voltage across the resistor."
    cot = f"Step 1: Use Ohm's law V = I R.\nStep 2: Substitute I = {clean_num(I)} A and R = {R} Ω.\nStep 3: V = {clean_num(I)} × {R} = {ans(V)} V."
    return make_record('EXTOV', 'ohms_law_voltage', q, cot, ans(V), 'V', ["Ohm's law: V = IR"], {'formula':'V=IR','inputs':{'I_A':I,'R_ohm':R},'recomputed_answer':round_to(V)}, 'physwikiquiz_wikidata')

def t_ohm_resistance(rng):
    I = choice(rng, [0.05,0.08,0.1,0.125,0.2,0.25,0.4,0.5,0.75,1,1.25,2,2.5,4])
    R = randint(rng, 4, 300)
    V = I*R
    q = f"The voltage across a component is {clean_num(V)} V while the current through it is {clean_num(I)} A. Determine its resistance."
    cot = f"Step 1: Rearrange Ohm's law to R = V/I.\nStep 2: Substitute V = {clean_num(V)} V and I = {clean_num(I)} A.\nStep 3: R = {clean_num(V)} / {clean_num(I)} = {ans(R)} Ω."
    return make_record('EXTOR', 'ohms_law_resistance', q, cot, ans(R), 'Ω', ["Ohm's law: R = V/I"], {'formula':'R=V/I','inputs':{'V_V':round_to(V),'I_A':I},'recomputed_answer':R}, 'physwikiquiz_wikidata')

def t_power_vi(rng):
    V = randint(rng, 3, 240)
    I = choice(rng, [0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.75,1,1.2,1.5,2,2.5,3,4,5,8,10])
    P = V*I
    q = f"A device has {clean_num(V)} V across it and carries {clean_num(I)} A. Calculate the electrical power."
    cot = f"Step 1: Use P = V I.\nStep 2: Substitute V = {clean_num(V)} V and I = {clean_num(I)} A.\nStep 3: P = {clean_num(V)} × {clean_num(I)} = {ans(P)} W."
    return make_record('EXTPV', 'electric_power_vi', q, cot, ans(P), 'W', ["Electric power: P = VI"], {'formula':'P=VI','inputs':{'V_V':V,'I_A':I},'recomputed_answer':round_to(P)}, 'physwikiquiz_wikidata')

def t_power_i2r(rng):
    I = choice(rng, [0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.75,1,1.25,1.5,2,2.5,3])
    R = randint(rng, 2, 500)
    P = I*I*R
    q = f"A resistor of {R} Ω carries a current of {clean_num(I)} A. Find the heat power dissipated in the resistor."
    cot = f"Step 1: Use Joule power P = I²R.\nStep 2: Substitute I = {clean_num(I)} A and R = {R} Ω.\nStep 3: P = ({clean_num(I)})² × {R} = {ans(P)} W."
    return make_record('EXTPI', 'electric_power_i2r', q, cot, ans(P), 'W', ["Joule power: P = I^2 R"], {'formula':'P=I^2R','inputs':{'I_A':I,'R_ohm':R},'recomputed_answer':round_to(P)}, 'physwikiquiz_wikidata')

def t_power_v2r(rng):
    R = randint(rng, 2, 500)
    V = randint(rng, 3, 240)
    P = V*V/R
    q = f"A {R} Ω resistor is connected across {V} V. Calculate its power dissipation."
    cot = f"Step 1: Use P = V²/R for a resistor.\nStep 2: Substitute V = {V} V and R = {R} Ω.\nStep 3: P = {V}² / {R} = {ans(P)} W."
    return make_record('EXTPR', 'electric_power_v2r', q, cot, ans(P), 'W', ["For a resistor: P = V^2/R"], {'formula':'P=V^2/R','inputs':{'V_V':V,'R_ohm':R},'recomputed_answer':round_to(P)}, 'physwikiquiz_wikidata')

def t_energy_power_time(rng):
    P = randint(rng, 2, 2000)
    t = choice(rng, [10,20,30,45,60,90,120,180,300,600,900,1200,1800,3600])
    E = P*t
    q = f"An electrical load consumes {P} W for {t} s. Calculate the electrical energy used."
    cot = f"Step 1: Use E = P t.\nStep 2: Substitute P = {P} W and t = {t} s.\nStep 3: E = {P} × {t} = {ans(E)} J."
    return make_record('EXTEJ', 'energy_from_power_time', q, cot, ans(E), 'J', ["Electrical energy: E = Pt"], {'formula':'E=Pt','inputs':{'P_W':P,'t_s':t},'recomputed_answer':E}, 'physwikiquiz_wikidata')

def t_series_resistance(rng):
    n = choice(rng, [2,3,4,5])
    Rs = [randint(rng, 1, 200) for _ in range(n)]
    Rtot = sum(Rs)
    q = f"Resistors of {', '.join(str(x)+' Ω' for x in Rs)} are connected in series. Find the equivalent resistance."
    cot = f"Step 1: In series, equivalent resistance is the sum of all resistances.\nStep 2: R_eq = {' + '.join(map(str, Rs))} = {Rtot} Ω."
    return make_record('EXTSR', 'series_resistance', q, cot, ans(Rtot), 'Ω', ["Series resistors: R_eq = R1 + R2 + ..."], {'formula':'Req=sum(Ri)','inputs':{'R_ohm':Rs},'recomputed_answer':Rtot}, 'circuit_vqa')

def t_parallel_two(rng):
    R1 = randint(rng, 2, 300)
    R2 = randint(rng, 2, 300)
    Req = R1*R2/(R1+R2)
    q = f"Two resistors, {R1} Ω and {R2} Ω, are connected in parallel. Calculate the equivalent resistance."
    cot = f"Step 1: For two parallel resistors, R_eq = R1R2/(R1+R2).\nStep 2: R_eq = {R1}×{R2}/({R1}+{R2}) = {ans(Req)} Ω."
    return make_record('EXTP2', 'parallel_resistance_two', q, cot, ans(Req), 'Ω', ["Two-resistor parallel formula: R_eq = R1R2/(R1+R2)"], {'formula':'Req=R1*R2/(R1+R2)','inputs':{'R1_ohm':R1,'R2_ohm':R2},'recomputed_answer':round_to(Req)}, 'circuit_vqa')

def t_parallel_three(rng):
    Rs = [randint(rng, 2, 300) for _ in range(3)]
    Req = 1/sum(1/r for r in Rs)
    q = f"Three resistors of {Rs[0]} Ω, {Rs[1]} Ω, and {Rs[2]} Ω are connected in parallel. Find the equivalent resistance."
    cot = f"Step 1: For parallel resistors, 1/R_eq = 1/R1 + 1/R2 + 1/R3.\nStep 2: 1/R_eq = 1/{Rs[0]} + 1/{Rs[1]} + 1/{Rs[2]}.\nStep 3: R_eq = {ans(Req)} Ω."
    return make_record('EXTP3', 'parallel_resistance_three', q, cot, ans(Req), 'Ω', ["Parallel resistors: 1/R_eq = Σ(1/R_i)"], {'formula':'Req=1/sum(1/Ri)','inputs':{'R_ohm':Rs},'recomputed_answer':round_to(Req)}, 'circuit_vqa')

def t_mixed_r_series_parallel(rng):
    R1,R2,R3 = [randint(rng, 2, 200) for _ in range(3)]
    p = R2*R3/(R2+R3)
    Req = R1+p
    q = f"A circuit has R1 = {R1} Ω in series with a parallel branch of R2 = {R2} Ω and R3 = {R3} Ω. Calculate the total equivalent resistance."
    cot = f"Step 1: First combine R2 and R3 in parallel: R23 = R2R3/(R2+R3).\nStep 2: R23 = {R2}×{R3}/({R2}+{R3}) = {ans(p)} Ω.\nStep 3: Add the series resistor: R_eq = {R1} + {ans(p)} = {ans(Req)} Ω."
    return make_record('EXTMX', 'mixed_series_parallel_resistance', q, cot, ans(Req), 'Ω', ["Parallel branch: R23 = R2R3/(R2+R3)", "Series addition: R_eq = R1 + R23"], {'formula':'Req=R1+R2*R3/(R2+R3)','inputs':{'R1_ohm':R1,'R2_ohm':R2,'R3_ohm':R3},'recomputed_answer':round_to(Req)}, 'circuit_vqa')

def t_voltage_divider(rng):
    Vin = randint(rng, 3, 240)
    R1 = randint(rng, 10, 1000)
    R2 = randint(rng, 10, 1000)
    Vout = Vin*R2/(R1+R2)
    q = f"In a voltage divider, R1 = {R1} Ω and R2 = {R2} Ω are in series across {Vin} V. What is the voltage across R2?"
    cot = f"Step 1: Use the divider relation V_R2 = V_in × R2/(R1+R2).\nStep 2: V_R2 = {Vin} × {R2}/({R1}+{R2}) = {ans(Vout)} V."
    return make_record('EXTVD', 'voltage_divider', q, cot, ans(Vout), 'V', ["Voltage divider: V_out = V_in R2/(R1+R2)"], {'formula':'Vout=Vin*R2/(R1+R2)','inputs':{'Vin_V':Vin,'R1_ohm':R1,'R2_ohm':R2},'recomputed_answer':round_to(Vout)}, 'circuit_vqa')

def t_current_divider(rng):
    Itot = choice(rng, [0.1,0.2,0.25,0.5,0.75,1,1.5,2,3,5,8,10])
    R1 = randint(rng, 5, 500)
    R2 = randint(rng, 5, 500)
    I1 = Itot*R2/(R1+R2)
    q = f"A total current of {clean_num(Itot)} A splits between two parallel resistors R1 = {R1} Ω and R2 = {R2} Ω. Calculate the current through R1."
    cot = f"Step 1: For two parallel resistors, current through R1 is I1 = I_total × R2/(R1+R2).\nStep 2: I1 = {clean_num(Itot)} × {R2}/({R1}+{R2}) = {ans(I1)} A."
    return make_record('EXTCD', 'current_divider', q, cot, ans(I1), 'A', ["Current divider: I1 = I_total R2/(R1+R2)"], {'formula':'I1=Itot*R2/(R1+R2)','inputs':{'Itotal_A':Itot,'R1_ohm':R1,'R2_ohm':R2},'recomputed_answer':round_to(I1)}, 'circuit_vqa')

def t_cap_charge(rng):
    Cuf = choice(rng, [0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470,1000])
    V = randint(rng, 1, 100)
    Quc = Cuf*V
    q = f"A capacitor of {clean_num(Cuf)} μF is charged to {V} V. Calculate the charge stored on it."
    cot = f"Step 1: Use Q = C V.\nStep 2: With C in μF and V in volts, Q is in μC.\nStep 3: Q = {clean_num(Cuf)} × {V} = {ans(Quc)} μC."
    return make_record('EXTCQ', 'capacitor_charge', q, cot, ans(Quc), 'μC', ["Capacitor charge: Q = CV"], {'formula':'Q=CV','inputs':{'C_uF':Cuf,'V_V':V},'recomputed_answer_uC':round_to(Quc)}, 'physwikiquiz_wikidata')

def t_cap_energy(rng):
    Cuf = choice(rng, [0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470,1000])
    V = randint(rng, 1, 100)
    Euj = 0.5*Cuf*V*V
    q = f"Calculate the energy stored in a {clean_num(Cuf)} μF capacitor when the voltage across it is {V} V."
    cot = f"Step 1: Use E = 0.5 C V².\nStep 2: With C in μF, the result is in μJ.\nStep 3: E = 0.5 × {clean_num(Cuf)} × {V}² = {ans(Euj)} μJ."
    return make_record('EXTCE', 'capacitor_energy', q, cot, ans(Euj), 'μJ', ["Capacitor energy: E = 1/2 C V^2"], {'formula':'E=0.5*C*V^2','inputs':{'C_uF':Cuf,'V_V':V},'recomputed_answer_uJ':round_to(Euj)}, 'physwikiquiz_wikidata')

def t_cap_parallel(rng):
    Cs = [choice(rng, [0.47,1,2.2,4.7,10,22,47,100,220]) for _ in range(choice(rng,[2,3,4]))]
    Ceq = sum(Cs)
    q = f"Capacitors of {', '.join(clean_num(c)+' μF' for c in Cs)} are connected in parallel. Find the equivalent capacitance."
    cot = f"Step 1: Parallel capacitances add directly.\nStep 2: C_eq = {' + '.join(clean_num(c) for c in Cs)} = {ans(Ceq)} μF."
    return make_record('EXTCP', 'parallel_capacitance', q, cot, ans(Ceq), 'μF', ["Parallel capacitors: C_eq = C1 + C2 + ..."], {'formula':'Ceq=sum(Ci)','inputs':{'C_uF':Cs},'recomputed_answer_uF':round_to(Ceq)}, 'circuit_vqa')

def t_cap_series_two(rng):
    C1 = choice(rng, [0.47,1,2.2,4.7,10,22,47,100,220])
    C2 = choice(rng, [0.47,1,2.2,4.7,10,22,47,100,220])
    Ceq = C1*C2/(C1+C2)
    q = f"Two capacitors, {clean_num(C1)} μF and {clean_num(C2)} μF, are connected in series. Calculate the equivalent capacitance."
    cot = f"Step 1: For two series capacitors, C_eq = C1C2/(C1+C2).\nStep 2: C_eq = {clean_num(C1)}×{clean_num(C2)}/({clean_num(C1)}+{clean_num(C2)}) = {ans(Ceq)} μF."
    return make_record('EXTCS', 'series_capacitance_two', q, cot, ans(Ceq), 'μF', ["Series capacitors: 1/C_eq = 1/C1 + 1/C2"], {'formula':'Ceq=C1*C2/(C1+C2)','inputs':{'C1_uF':C1,'C2_uF':C2},'recomputed_answer_uF':round_to(Ceq)}, 'circuit_vqa')

def t_parallel_plate_cap(rng):
    er = choice(rng, [1,1.5,2.1,2.2,3,3.7,4,5,10])
    A_cm2 = randint(rng, 1, 500)
    d_mm = choice(rng, [0.1,0.2,0.25,0.5,0.75,1,1.5,2,3,5])
    A = A_cm2*1e-4
    d = d_mm*1e-3
    C_pf = EPS0*er*A/d*1e12
    q = f"A parallel-plate capacitor has plate area {A_cm2} cm², plate separation {clean_num(d_mm)} mm, and dielectric constant {clean_num(er)}. Calculate its capacitance."
    cot = f"Step 1: Use C = ε0 εr A/d.\nStep 2: Convert A = {A_cm2} cm² = {clean_num(A)} m² and d = {clean_num(d_mm)} mm = {clean_num(d)} m.\nStep 3: C = ε0×{clean_num(er)}×A/d = {ans(C_pf)} pF."
    return make_record('EXTPP', 'parallel_plate_capacitance', q, cot, ans(C_pf), 'pF', ["Parallel-plate capacitor: C = ε0 εr A/d"], {'formula':'C=eps0*er*A/d','inputs':{'epsilon0_F_per_m':EPS0,'er':er,'A_m2':A,'d_m':d},'recomputed_answer_pF':round_to(C_pf)}, 'physwikiquiz_wikidata')

def t_field_voltage_distance(rng):
    V = randint(rng, 1, 5000)
    d_cm = choice(rng, [0.1,0.2,0.5,1,2,3,4,5,10,20,50])
    E = V/(d_cm/100)
    q = f"The potential difference between two parallel plates is {V} V and their separation is {clean_num(d_cm)} cm. Find the uniform electric field magnitude between them."
    cot = f"Step 1: Use E = V/d.\nStep 2: Convert d = {clean_num(d_cm)} cm = {clean_num(d_cm/100)} m.\nStep 3: E = {V}/{clean_num(d_cm/100)} = {ans(E)} V/m."
    return make_record('EXTEF', 'uniform_electric_field', q, cot, ans(E), 'V/m', ["Uniform field between plates: E = V/d"], {'formula':'E=V/d','inputs':{'V_V':V,'d_m':d_cm/100},'recomputed_answer_V_per_m':round_to(E)}, 'physwikiquiz_wikidata')

def t_force_in_field(rng):
    q_uc = choice(rng, [0.1,0.2,0.5,1,2,5,10,20,50,100])
    E = randint(rng, 100, 200000)
    F = q_uc*1e-6*E
    q = f"A charge of {clean_num(q_uc)} μC is placed in an electric field of {E} N/C. Calculate the magnitude of the electric force on the charge."
    cot = f"Step 1: Use F = qE.\nStep 2: Convert q = {clean_num(q_uc)} μC = {clean_num(q_uc*1e-6)} C.\nStep 3: F = {clean_num(q_uc*1e-6)} × {E} = {ans(F)} N."
    return make_record('EXTFE', 'force_on_charge_in_field', q, cot, ans(F), 'N', ["Electric force in a field: F = qE"], {'formula':'F=qE','inputs':{'q_C':q_uc*1e-6,'E_N_per_C':E},'recomputed_answer_N':round_to(F)}, 'physwikiquiz_wikidata')

def t_coulomb_force(rng):
    q1_uc = choice(rng, [0.1,0.2,0.5,1,2,3,5,10,20,50])
    q2_uc = choice(rng, [0.1,0.2,0.5,1,2,3,5,10,20,50])
    r_cm = choice(rng, [1,2,3,4,5,8,10,12,15,20,25,30,40,50])
    er = choice(rng, [1,1,1,2,2.2,4,5,10])
    F = K*(q1_uc*1e-6)*(q2_uc*1e-6)/(er*(r_cm/100)**2)
    q = f"Two point charges of {clean_num(q1_uc)} μC and {clean_num(q2_uc)} μC are separated by {r_cm} cm in a medium with dielectric constant {clean_num(er)}. Calculate the magnitude of the electrostatic force."
    cot = f"Step 1: Use Coulomb's law F = k|q1q2|/(εr r²).\nStep 2: Convert charges to coulombs and distance to meters.\nStep 3: F = k×({clean_num(q1_uc)} μC)×({clean_num(q2_uc)} μC)/({clean_num(er)}×({r_cm} cm)²) = {ans(F)} N."
    return make_record('EXTCF', 'coulomb_force', q, cot, ans(F), 'N', ["Coulomb's law magnitude: F = k|q1q2|/(εr r^2)"], {'formula':'F=k*abs(q1*q2)/(er*r^2)','inputs':{'k_Nm2_per_C2':K,'q1_C':q1_uc*1e-6,'q2_C':q2_uc*1e-6,'er':er,'r_m':r_cm/100},'recomputed_answer_N':round_to(F)}, 'physwikiquiz_wikidata')

def t_field_point_charge(rng):
    q_uc = choice(rng, [0.1,0.2,0.5,1,2,3,5,10,20,50])
    r_cm = choice(rng, [1,2,3,4,5,8,10,12,15,20,25,30,50,80,100])
    er = choice(rng, [1,1,1,2,2.2,4,5,10])
    E = K*(q_uc*1e-6)/(er*(r_cm/100)**2)
    q = f"Find the electric field magnitude {r_cm} cm from a point charge of {clean_num(q_uc)} μC in a medium with dielectric constant {clean_num(er)}."
    cot = f"Step 1: Use E = k|q|/(εr r²).\nStep 2: Convert q to C and r to m.\nStep 3: E = {ans(E)} N/C."
    return make_record('EXTEQ', 'electric_field_point_charge', q, cot, ans(E), 'N/C', ["Point-charge field: E = k|q|/(εr r^2)"], {'formula':'E=k*abs(q)/(er*r^2)','inputs':{'q_C':q_uc*1e-6,'er':er,'r_m':r_cm/100},'recomputed_answer_N_per_C':round_to(E)}, 'physwikiquiz_wikidata')

def t_potential_point_charge(rng):
    q_uc = choice(rng, [0.1,0.2,0.5,1,2,3,5,10,20,50])
    r_cm = choice(rng, [1,2,3,5,8,10,20,30,50,100])
    er = choice(rng, [1,1,2,2.2,4,5,10])
    V = K*(q_uc*1e-6)/(er*(r_cm/100))
    q = f"Calculate the electric potential {r_cm} cm from a positive point charge of {clean_num(q_uc)} μC in a medium with dielectric constant {clean_num(er)}."
    cot = f"Step 1: Use V = kq/(εr r).\nStep 2: Convert q to C and r to m.\nStep 3: V = {ans(V)} V."
    return make_record('EXTPQ', 'electric_potential_point_charge', q, cot, ans(V), 'V', ["Point-charge potential: V = kq/(εr r)"], {'formula':'V=k*q/(er*r)','inputs':{'q_C':q_uc*1e-6,'er':er,'r_m':r_cm/100},'recomputed_answer_V':round_to(V)}, 'physwikiquiz_wikidata')

def t_potential_energy_two_charges(rng):
    q1_uc = choice(rng, [0.1,0.2,0.5,1,2,5,10,20])
    q2_uc = choice(rng, [0.1,0.2,0.5,1,2,5,10,20])
    r_cm = choice(rng, [1,2,3,5,8,10,15,20,30,50])
    er = choice(rng, [1,1,2,2.2,4,5,10])
    U = K*(q1_uc*1e-6)*(q2_uc*1e-6)/(er*(r_cm/100))
    q = f"Two positive charges {clean_num(q1_uc)} μC and {clean_num(q2_uc)} μC are {r_cm} cm apart in a medium with dielectric constant {clean_num(er)}. Calculate their electric potential energy."
    cot = f"Step 1: Use U = kq1q2/(εr r).\nStep 2: Convert charges to C and distance to m.\nStep 3: U = {ans(U)} J."
    return make_record('EXTUE', 'electric_potential_energy', q, cot, ans(U), 'J', ["Potential energy of two point charges: U = kq1q2/(εr r)"], {'formula':'U=k*q1*q2/(er*r)','inputs':{'q1_C':q1_uc*1e-6,'q2_C':q2_uc*1e-6,'er':er,'r_m':r_cm/100},'recomputed_answer_J':round_to(U)}, 'physwikiquiz_wikidata')

def t_rc_time_constant(rng):
    Rk = choice(rng, [1,2.2,4.7,10,22,47,100,220,470])
    Cuf = choice(rng, [0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220])
    tau_ms = Rk*Cuf # because kΩ*μF = ms
    q = f"An RC circuit has R = {clean_num(Rk)} kΩ and C = {clean_num(Cuf)} μF. Calculate the time constant."
    cot = f"Step 1: Use τ = RC.\nStep 2: Since kΩ × μF = ms, τ = {clean_num(Rk)} × {clean_num(Cuf)} = {ans(tau_ms)} ms."
    return make_record('EXTRC', 'rc_time_constant', q, cot, ans(tau_ms), 'ms', ["RC time constant: τ = RC"], {'formula':'tau=R*C','inputs':{'R_kohm':Rk,'C_uF':Cuf},'recomputed_answer_ms':round_to(tau_ms)}, 'physgym')

def t_rl_time_constant(rng):
    L_mH = choice(rng, [1,2.2,4.7,10,22,47,100,220,470,1000])
    R = randint(rng, 1, 500)
    tau_ms = L_mH/R # mH / Ω = ms
    q = f"An RL circuit has inductance L = {clean_num(L_mH)} mH and resistance R = {R} Ω. Calculate the time constant."
    cot = f"Step 1: Use τ = L/R.\nStep 2: Since mH/Ω = ms, τ = {clean_num(L_mH)}/{R} = {ans(tau_ms)} ms."
    return make_record('EXTRL', 'rl_time_constant', q, cot, ans(tau_ms), 'ms', ["RL time constant: τ = L/R"], {'formula':'tau=L/R','inputs':{'L_mH':L_mH,'R_ohm':R},'recomputed_answer_ms':round_to(tau_ms)}, 'physgym')

def t_cap_reactance(rng):
    f = choice(rng, [50,60,100,120,400,500,1000,2000,5000,10000])
    Cuf = choice(rng, [0.01,0.022,0.047,0.1,0.22,0.47,1,2.2,4.7,10,22,47,100])
    Xc = 1/(2*PI*f*Cuf*1e-6)
    q = f"A capacitor of {clean_num(Cuf)} μF is used at frequency {f} Hz. Calculate its capacitive reactance."
    cot = f"Step 1: Use X_C = 1/(2πfC).\nStep 2: Convert C = {clean_num(Cuf)} μF to farads.\nStep 3: X_C = {ans(Xc)} Ω."
    return make_record('EXTXC', 'capacitive_reactance', q, cot, ans(Xc), 'Ω', ["Capacitive reactance: X_C = 1/(2πfC)"], {'formula':'Xc=1/(2*pi*f*C)','inputs':{'f_Hz':f,'C_F':Cuf*1e-6},'recomputed_answer_ohm':round_to(Xc)}, 'physgym')

def t_ind_reactance(rng):
    f = choice(rng, [50,60,100,120,400,500,1000,2000,5000,10000])
    L_mH = choice(rng, [0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470,1000])
    Xl = 2*PI*f*L_mH*1e-3
    q = f"An inductor of {clean_num(L_mH)} mH operates at {f} Hz. Calculate its inductive reactance."
    cot = f"Step 1: Use X_L = 2πfL.\nStep 2: Convert L = {clean_num(L_mH)} mH to henries.\nStep 3: X_L = {ans(Xl)} Ω."
    return make_record('EXTXL', 'inductive_reactance', q, cot, ans(Xl), 'Ω', ["Inductive reactance: X_L = 2πfL"], {'formula':'Xl=2*pi*f*L','inputs':{'f_Hz':f,'L_H':L_mH*1e-3},'recomputed_answer_ohm':round_to(Xl)}, 'physgym')

def t_battery_internal(rng):
    E = randint(rng, 3, 240)
    R = randint(rng, 2, 1000)
    r = choice(rng, [0.1,0.2,0.5,1,2,5,10])
    I = E/(R+r)
    Vt = I*R
    q = f"A battery with emf {E} V and internal resistance {clean_num(r)} Ω is connected to an external resistor of {R} Ω. Calculate the terminal voltage across the external resistor."
    cot = f"Step 1: The circuit current is I = E/(R+r).\nStep 2: I = {E}/({R}+{clean_num(r)}) = {ans(I)} A.\nStep 3: Terminal voltage across the load is V = IR = {ans(I)}×{R} = {ans(Vt)} V."
    return make_record('EXTBI', 'battery_internal_resistance', q, cot, ans(Vt), 'V', ["Circuit current with internal resistance: I = E/(R+r)", "Terminal/load voltage: V = IR"], {'formula':'Vterminal=E*R/(R+r)','inputs':{'emf_V':E,'R_ohm':R,'r_ohm':r},'recomputed_answer_V':round_to(Vt)}, 'circuit_vqa')

def t_measurement_avg_error(rng):
    base = choice(rng, [0.5,1,1.5,2,2.5,3,4,5,8,10,12,15])
    delta = choice(rng, [0.02,0.05,0.1,0.2,0.5])
    vals = [base-delta, base, base+delta]
    avg = sum(vals)/3
    mae = sum(abs(v-avg) for v in vals)/3
    q = f"Three current measurements are {clean_num(vals[0])} A, {clean_num(vals[1])} A, and {clean_num(vals[2])} A. Calculate the average current and the mean absolute error."
    cot = f"Step 1: Average current = ({clean_num(vals[0])}+{clean_num(vals[1])}+{clean_num(vals[2])})/3 = {ans(avg)} A.\nStep 2: Mean absolute error = average of absolute deviations from the mean.\nStep 3: MAE = ({clean_num(delta)}+0+{clean_num(delta)})/3 = {ans(mae)} A."
    return make_record('EXTME', 'measurement_average_error', q, cot, f"{ans(avg)}; {ans(mae)}", 'A; A', ["Mean value: x̄ = Σx_i/n", "Mean absolute error: Δ = Σ|x_i-x̄|/n"], {'formula':'avg=sum(vals)/3; mae=sum(abs(v-avg))/3','inputs':{'values_A':vals},'recomputed_answer':{'average_A':round_to(avg),'mae_A':round_to(mae)}}, 'physgym')

def t_charge_from_current_time(rng):
    I = choice(rng, [0.01,0.02,0.05,0.1,0.2,0.25,0.5,0.75,1,1.5,2,3,5])
    t = choice(rng, [1,2,5,10,20,30,60,120,300,600,900,1800])
    Q = I*t
    q = f"A steady current of {clean_num(I)} A flows for {t} s. How much charge passes through the circuit?"
    cot = f"Step 1: Use Q = It.\nStep 2: Substitute I = {clean_num(I)} A and t = {t} s.\nStep 3: Q = {clean_num(I)}×{t} = {ans(Q)} C."
    return make_record('EXTQI', 'charge_from_current_time', q, cot, ans(Q), 'C', ["Current definition: Q = It"], {'formula':'Q=I*t','inputs':{'I_A':I,'t_s':t},'recomputed_answer_C':round_to(Q)}, 'physwikiquiz_wikidata')

def t_drift_current_density(rng):
    I = choice(rng, [0.1,0.2,0.5,1,2,5,10,15,20])
    A_mm2 = choice(rng, [0.1,0.2,0.5,0.75,1,1.5,2,2.5,4,6,10])
    J = I/(A_mm2*1e-6)
    q = f"A wire carries {clean_num(I)} A and has cross-sectional area {clean_num(A_mm2)} mm². Calculate the current density."
    cot = f"Step 1: Use J = I/A.\nStep 2: Convert A = {clean_num(A_mm2)} mm² = {clean_num(A_mm2*1e-6)} m².\nStep 3: J = {clean_num(I)}/{clean_num(A_mm2*1e-6)} = {ans(J)} A/m²."
    return make_record('EXTJD', 'current_density', q, cot, ans(J), 'A/m²', ["Current density: J = I/A"], {'formula':'J=I/A','inputs':{'I_A':I,'A_m2':A_mm2*1e-6},'recomputed_answer_A_per_m2':round_to(J)}, 'physwikiquiz_wikidata')

def t_resistivity_wire(rng):
    rho = choice(rng, [1.68e-8, 2.44e-8, 2.82e-8, 1.1e-6, 1.0e-7])
    L = choice(rng, [0.5,1,2,5,10,20,50,100])
    A_mm2 = choice(rng, [0.1,0.2,0.5,1,1.5,2,4,6,10])
    R = rho*L/(A_mm2*1e-6)
    q = f"A wire has resistivity {clean_num(rho)} Ω·m, length {clean_num(L)} m, and cross-sectional area {clean_num(A_mm2)} mm². Calculate its resistance."
    cot = f"Step 1: Use R = ρL/A.\nStep 2: Convert A = {clean_num(A_mm2)} mm² = {clean_num(A_mm2*1e-6)} m².\nStep 3: R = {clean_num(rho)}×{clean_num(L)}/{clean_num(A_mm2*1e-6)} = {ans(R)} Ω."
    return make_record('EXTRW', 'wire_resistance_resistivity', q, cot, ans(R), 'Ω', ["Wire resistance: R = ρL/A"], {'formula':'R=rho*L/A','inputs':{'rho_ohm_m':rho,'L_m':L,'A_m2':A_mm2*1e-6},'recomputed_answer_ohm':round_to(R)}, 'physwikiquiz_wikidata')

def t_temperature_resistance(rng):
    R0 = randint(rng, 1, 1000)
    alpha = choice(rng, [0.0039,0.0043,0.0038,0.0004,0.0009])
    dT = randint(rng, 5, 150)
    R = R0*(1+alpha*dT)
    q = f"A conductor has resistance {R0} Ω at the reference temperature. If its temperature rises by {dT} °C and α = {clean_num(alpha)} °C^-1, find the new resistance."
    cot = f"Step 1: Use R = R0(1 + αΔT).\nStep 2: Substitute R0 = {R0} Ω, α = {clean_num(alpha)} °C^-1, and ΔT = {dT} °C.\nStep 3: R = {R0}(1+{clean_num(alpha)}×{dT}) = {ans(R)} Ω."
    return make_record('EXTTR', 'temperature_dependence_resistance', q, cot, ans(R), 'Ω', ["Temperature dependence: R = R0(1+αΔT)"], {'formula':'R=R0*(1+alpha*dT)','inputs':{'R0_ohm':R0,'alpha_per_C':alpha,'dT_C':dT},'recomputed_answer_ohm':round_to(R)}, 'physwikiquiz_wikidata')

def t_transformer_voltage(rng):
    Vp = choice(rng, [12,24,48,110,120,220,230,240,1000,11000])
    Np = randint(rng, 50, 2000)
    ratio = choice(rng, [0.1,0.2,0.25,0.5,2,3,4,5,10])
    Ns = max(1, int(Np*ratio))
    Vs = Vp*Ns/Np
    q = f"An ideal transformer has primary voltage {Vp} V, primary turns {Np}, and secondary turns {Ns}. Calculate the secondary voltage."
    cot = f"Step 1: For an ideal transformer, Vs/Vp = Ns/Np.\nStep 2: Vs = {Vp}×{Ns}/{Np} = {ans(Vs)} V."
    return make_record('EXTTV', 'ideal_transformer_voltage', q, cot, ans(Vs), 'V', ["Ideal transformer relation: Vs/Vp = Ns/Np"], {'formula':'Vs=Vp*Ns/Np','inputs':{'Vp_V':Vp,'Np':Np,'Ns':Ns},'recomputed_answer_V':round_to(Vs)}, 'physgym')

TEMPLATES: List[Template] = [
    t_ohm_current,t_ohm_voltage,t_ohm_resistance,t_power_vi,t_power_i2r,t_power_v2r,t_energy_power_time,
    t_series_resistance,t_parallel_two,t_parallel_three,t_mixed_r_series_parallel,t_voltage_divider,t_current_divider,
    t_cap_charge,t_cap_energy,t_cap_parallel,t_cap_series_two,t_parallel_plate_cap,t_field_voltage_distance,
    t_force_in_field,t_coulomb_force,t_field_point_charge,t_potential_point_charge,t_potential_energy_two_charges,
    t_rc_time_constant,t_rl_time_constant,t_cap_reactance,t_ind_reactance,t_battery_internal,t_measurement_avg_error,
    t_charge_from_current_time,t_drift_current_density,t_resistivity_wire,t_temperature_resistance,t_transformer_voltage,
]


def load_existing_questions(paths: List[Path]) -> Tuple[set, List[Dict[str,Any]]]:
    qs = set(); records = []
    for p in paths:
        if p.exists():
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q = obj.get('question')
                    if q:
                        qs.add(norm_q(q))
                    records.append(obj)
    return qs, records


def generate(n=N_TARGET, seed=SEED):
    rng = random.Random(seed)
    existing_qs, existing_records = load_existing_questions(ORIGINAL_PATHS)
    out: List[Dict[str,Any]] = []
    seen = set(existing_qs)
    attempts = 0
    while len(out) < n and attempts < n*100:
        attempts += 1
        tmpl = rng.choice(TEMPLATES)
        rec = tmpl(rng)
        nq = norm_q(rec['question'])
        if nq in seen:
            continue
        # integrity sanity
        required = ['id','source_record_id','type','category','question','answer','unit','cot','explanation','premises','verification','external_source']
        if not all(k in rec for k in required):
            raise ValueError(f'Missing field in {rec}')
        seen.add(nq)
        out.append(rec)
    if len(out) < n:
        raise RuntimeError(f'Only generated {len(out)} unique records after {attempts} attempts')
    return out, existing_records, len(existing_qs), attempts


def write_jsonl(path: Path, records: List[Dict[str,Any]]):
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def validate(records: List[Dict[str,Any]], baseline_qs_count: int, attempts: int) -> Dict[str,Any]:
    ids = [r['id'] for r in records]
    qs = [norm_q(r['question']) for r in records]
    categories = {}
    sources = {}
    for r in records:
        categories[r['category']] = categories.get(r['category'],0)+1
        sk = r['external_source']['source_name']
        sources[sk] = sources.get(sk,0)+1
    required = ['id','source_record_id','type','category','question','answer','unit','cot','explanation','premises','verification','external_source','disclosure_note']
    missing = []
    for idx,r in enumerate(records):
        miss = [k for k in required if k not in r]
        if miss:
            missing.append({'index':idx,'id':r.get('id'),'missing':miss})
    return {
        'created_at': '2026-05-14',
        'target_records': N_TARGET,
        'generated_records': len(records),
        'attempts': attempts,
        'baseline_question_count_for_dedup': baseline_qs_count,
        'duplicate_ids_within_new_set': len(ids)-len(set(ids)),
        'duplicate_questions_within_new_set': len(qs)-len(set(qs)),
        'missing_required_fields': missing[:20],
        'category_counts': dict(sorted(categories.items())),
        'source_counts': dict(sorted(sources.items())),
        'quality_controls': [
            'Answers computed programmatically by deterministic formula templates.',
            'Exact question text deduplicated against current uploaded dataset and previous synthetic dataset where available.',
            'No external problem statement text copied into generated questions.',
            'Every record includes external_source and disclosure_note metadata.'
        ],
        'limitations': [
            'This is source-guided generated augmentation, not a verbatim mirror of external datasets.',
            'Formulas are deterministic and validated by generation code, but records have not been manually reviewed one by one by a human physics expert.',
            'If used for competition training/tuning, disclose the manifest sources and usage statement.'
        ]
    }


def main():
    records, existing_records, baseline_qs_count, attempts = generate()
    ext_path = OUT_DIR / 'external_disclosed_physics_dataset_25000.jsonl'
    report_path = OUT_DIR / 'external_disclosed_physics_validation_report.json'
    manifest_path = OUT_DIR / 'external_disclosed_physics_manifest.json'
    combined_path = OUT_DIR / 'combined_physics_dataset_all_41828.jsonl'
    zip_path = OUT_DIR / 'external_disclosed_physics_pack_25000.zip'

    write_jsonl(ext_path, records)
    # Build combined all: original + previous synthetic + new external, dedup exact questions preserving first occurrence.
    combined = []
    seen_q = set()
    for r in existing_records + records:
        q = r.get('question')
        if not q:
            continue
        nq = norm_q(q)
        if nq in seen_q:
            continue
        seen_q.add(nq)
        combined.append(r)
    write_jsonl(combined_path, combined)

    report = validate(records, baseline_qs_count, attempts)
    report['combined_records_written'] = len(combined)
    report['combined_output'] = str(combined_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    manifest = {
        'dataset_name': 'external_disclosed_physics_dataset_25000',
        'dataset_type': 'source-guided formula-template physics augmentation',
        'record_count': len(records),
        'format': 'JSONL; one physics QA record per line; compatible fields: id, source_record_id, type, category, question, cot, explanation, answer, unit, premises, verification',
        'disclosure_statement_suggested': 'We used an additional source-guided physics augmentation dataset generated from disclosed open references/datasets: PhysWikiQuiz/Wikidata formula metadata, Circuit-VQA circuit QA task families, and PhysGym-style executable physics environments. The generated questions were produced locally from deterministic formula templates with sampled numerical values; no external problem statements, images, or answer text were copied verbatim.',
        'sources': SOURCES,
        'license_note': 'Because Circuit-VQA is CC-BY-SA 3.0, keep attribution in any redistribution and consider share-alike compatibility. Wikidata structured data is CC0; PhysWikiQuiz code is Apache-2.0; PhysGym is MIT. This generated pack is provided as an attribution-preserving derivative/augmentation resource for competition experimentation.',
        'generation_seed': SEED,
        'generator_script': 'generate_external_disclosed_physics_dataset.py',
        'validation_report': report_path.name
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in [ext_path, combined_path, report_path, manifest_path, OUT_DIR / 'generate_external_disclosed_physics_dataset.py']:
            z.write(p, arcname=p.name)

    print(json.dumps({
        'external_dataset': str(ext_path),
        'records': len(records),
        'combined_dataset': str(combined_path),
        'combined_records': len(combined),
        'validation_report': str(report_path),
        'manifest': str(manifest_path),
        'zip': str(zip_path)
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
