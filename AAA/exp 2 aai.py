
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# IMPORTANT:
# This import avoids the statsmodels import issue

from pgmpy.inference.ExactInference import VariableElimination

model = DiscreteBayesianNetwork([
    ('StudyHours', 'Pass'),
    ('Attendance', 'Pass'),
    ('InternalMarks', 'Pass')
])

cpd_study = TabularCPD(
    variable='StudyHours',
    variable_card=2,
    values=[
        [0.4],   # Low
        [0.6]    # High
    ]
)

cpd_attendance = TabularCPD(
    variable='Attendance',
    variable_card=2,
    values=[
        [0.3],   # Irregular
        [0.7]    # Regular
    ]
)

cpd_internal = TabularCPD(
    variable='InternalMarks',
    variable_card=2,
    values=[
        [0.5],   # Low
        [0.5]    # High
    ]
)

cpd_pass = TabularCPD(
    variable='Pass',
    variable_card=2,

    values=[

        # FAIL probabilities
        [
            0.9, 0.7, 0.6, 0.4,
            0.6, 0.3, 0.2, 0.1
        ],

        # PASS probabilities
        [
            0.1, 0.3, 0.4, 0.6,
            0.4, 0.7, 0.8, 0.9
        ]
    ],

    evidence=[
        'StudyHours',
        'Attendance',
        'InternalMarks'
    ],

    evidence_card=[2, 2, 2]
)

model.add_cpds(
    cpd_study,
    cpd_attendance,
    cpd_internal,
    cpd_pass
)

print("================================================")
print("Checking Bayesian Network Model")
print("================================================")

print("Model Valid :", model.check_model())


inference = VariableElimination(model)

# ------------------------------------------------------------
# Query:
#
# StudyHours = High (1)
# Attendance = Regular (1)
# InternalMarks = High (1)
# ------------------------------------------------------------

result = inference.query(
    variables=['Pass'],

    evidence={
        'StudyHours': 1,
        'Attendance': 1,
        'InternalMarks': 1
    }
)

# ============================================================
# STEP 8 : Display Result
# ============================================================

print("\n================================================")
print("Probability of Student Passing Exam")
print("================================================\n")

print(result)

# ============================================================
# STEP 9 : Conclusion
# ============================================================

print("\n================================================")
print("Conclusion")
print("================================================")

print("""
The Bayesian Network was successfully implemented
to predict student exam performance.

The prediction depends on:
1. Study Hours
2. Attendance
3. Internal Marks

The network uses Conditional Probability Distributions
(CPDs) and probabilistic inference to determine the
likelihood of passing the exam.

The model shows that when:
- Study Hours are High
- Attendance is Regular
- Internal Marks are High

the probability of passing becomes very high.
""")