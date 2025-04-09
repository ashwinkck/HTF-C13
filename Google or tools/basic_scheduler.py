from ortools.sat.python import cp_model

# 📦 Sample employee data
employees = [
    {"id": "Alice", "isAvailable": True, "skills": ["python"]},
    {"id": "Bob", "isAvailable": False, "skills": ["ml"]},
    {"id": "Charlie", "isAvailable": True, "skills": ["python", "ml"]}
]

# 📦 Sample task data
tasks = [
    {"id": "Task1", "base_duration": 1, "required_skills": ["python"]},
    {"id": "Task2", "base_duration": 1, "required_skills": ["ml"]},
    {"id": "Task3", "base_duration": 1, "required_skills": ["python"]}
]

# ⏰ Working hours from 9AM to 5PM
time_slots = list(range(9, 18))  # 9 to 17

# 🔮 Duration predictor (currently just returns base)
def predict_duration(task, employee):
    return task["base_duration"]

# 📐 Create CP-SAT model
model = cp_model.CpModel()

# 📌 Variables
assignments = {}  # Map (task, employee, start_time) -> decision var

for t, task in enumerate(tasks):
    for e, emp in enumerate(employees):
        # Skip if employee is unavailable or lacks required skills
        if not emp["isAvailable"]:
            continue
        if not all(skill in emp["skills"] for skill in task["required_skills"]):
            continue

        for start_time in time_slots:
            end_time = start_time + predict_duration(task, emp)
            if end_time <= max(time_slots):
                var = model.NewBoolVar(f"task{t}_emp{e}_start{start_time}")
                assignments[(t, e, start_time)] = var

# 📏 Constraint 1: Each task assigned exactly once
for t in range(len(tasks)):
    model.AddExactlyOne(assignments.get((t, e, s), 0) for e in range(len(employees)) for s in time_slots)

# 📏 Constraint 2: No employee does 2 tasks at once
for e in range(len(employees)):
    if not employees[e]["isAvailable"]:
        continue
    for s in time_slots:
        model.AddAtMostOne(assignments.get((t, e, s), 0) for t in range(len(tasks)))

# 🎯 Objective: Maximize number of scheduled tasks
model.Maximize(sum(assignments.values()))

# 🚀 Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

# 📤 Output
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("\nSchedule:")
    schedule = []
    for (t, e, s), var in assignments.items():
        if solver.Value(var) == 1:
            task = tasks[t]
            emp = employees[e]
            duration = predict_duration(task, emp)
            schedule.append({
                "employeeId": emp["id"],
                "taskId": task["id"],
                "startTime": s,
                "endTime": s + duration
            })
    for entry in schedule:
        print(entry)
else:
    print("No valid schedule found.")
