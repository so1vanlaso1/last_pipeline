# EXACT 2026 Physics Formula-Verified Test Dataset (484 records)

Purpose: independent local test set for evaluating coverage of physics QA models targeting the EXACT 2026 physics track.

Important limitation: this is not the official hidden test set and does not guarantee which questions will appear in the competition. It is formula-verified against deterministic templates covering the stated physics topic scope.

Files:
- `exact2026_physics_formula_verified_test_484_compatible.jsonl`: same core schema as the uploaded train set.
- `exact2026_physics_formula_verified_test_484_full.jsonl`: includes category, premises, and verification metadata.
- CSV versions of both files.
- Report JSON with coverage counts.

Schema compatible fields:
`id`, `source_record_id`, `type`, `question`, `answer`, `unit`, `cot`, `explanation`.

Formula assumptions:
- Coulomb constant k = 9 × 10^9 N·m²/C².
- Vacuum permittivity epsilon_0 = 8.854 × 10^-12 F/m.
- Air is approximated as vacuum for electrostatics templates.
- Numerical answers are rounded compactly to at most 3 decimals when needed.

Coverage:
- Ohm's law: current, voltage, resistance.
- Series/parallel resistors and total current.
- Voltage dividers.
- Resistor power.
- Charge-current-time relation.
- Capacitor charge, voltage, energy.
- Series/parallel capacitors.
- Parallel-plate capacitance.
- Uniform electric field between plates.
- Coulomb force, point-charge electric field, point-charge electric potential.
- Potential energy, force on charge in uniform field.
- RC time constant.

Generation approach: questions are original synthetic variants generated from public physics formulas, not copied verbatim from online problem sets.
