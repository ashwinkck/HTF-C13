import json
from ortools.sat.python import cp_model

# Load your data
with open(r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\Google or tools\dataset\datafiveempsixtask.json") as f:
    data = json.load(f)

employees = data["employees"]
tasks = data["tasks"]

priority_map = {"high": 0, "medium": 1, "low": 2}
model = cp_model.CpModel()  
num_days = 30

task_vars = {}
penalties = []

for task in tasks:
    task_id = task["taskId"]
    duration = max(1, task.get("estimatedDurationDays", task.get("estimatedDurationHours", 8) // 8))

    for emp in employees:
        emp_id = emp["empId"]
        start_day = model.NewIntVar(0, num_days - duration, f"start_{task_id}_{emp_id}")
        assigned = model.NewBoolVar(f"assigned_{task_id}_{emp_id}")
        task_vars[(task_id, emp_id)] = (start_day, assigned)

# Constraints
for task in tasks:
    task_id = task["taskId"]
    required_skills = {s["skillName"]: s["minLevel"] for s in task.get("requiredSkills", [])}
    duration = max(1, task.get("estimatedDurationDays", task.get("estimatedDurationHours", 8) // 8))

    assigned_vars = []

    for emp in employees:
        emp_id = emp["empId"]
        skills = {s["skillName"]: s["level"] for s in emp.get("skills", [])}
        availability = set()
        for pattern in emp.get("availabilityPatterns", []):
            if pattern["type"] == "weekly":
                availability.update(pattern["days"])

        start_day, assigned = task_vars[(task_id, emp_id)]
        assigned_vars.append(assigned)

        # Skill constraints
        for skill, min_level in required_skills.items():
            if skill not in skills or skills[skill] < min_level:
                model.Add(assigned == 0)

        # Availability constraints
        for d in range(num_days - duration + 1):
            if not all((d + offset) % 7 in availability for offset in range(duration)):
                model.Add(start_day != d).OnlyEnforceIf(assigned)

    # Soft constraint: allow zero or one assignment with penalty if none
    penalty = model.NewBoolVar(f"unassigned_{task_id}")
    model.Add(sum(assigned_vars) <= 1)
    model.Add(sum(assigned_vars) == 0).OnlyEnforceIf(penalty)
    model.Add(sum(assigned_vars) >= 1).OnlyEnforceIf(penalty.Not())
    penalties.append(penalty)

# No overlapping tasks for employees
for emp in employees:
    emp_id = emp["empId"]
    for day in range(num_days):
        active_tasks = []
        for task in tasks:
            task_id = task["taskId"]
            duration = max(1, task.get("estimatedDurationDays", task.get("estimatedDurationHours", 8) // 8))
            start_day, assigned = task_vars[(task_id, emp_id)]
            in_day = model.NewBoolVar(f"{task_id}_{emp_id}_on_day_{day}")

            model.Add(start_day <= day).OnlyEnforceIf(in_day)
            model.Add(day < start_day + duration).OnlyEnforceIf(in_day)
            model.AddImplication(in_day, assigned)
            active_tasks.append(in_day)

        model.Add(sum(active_tasks) <= 1)

# Dependencies
task_id_map = {t["taskId"]: t for t in tasks}
for task in tasks:
    for dep_id in task.get("dependencies", []):
        dep_task = task_id_map[dep_id]
        dep_duration = max(1, dep_task.get("estimatedDurationDays", dep_task.get("estimatedDurationHours", 8) // 8))
        for emp1 in employees:
            for emp2 in employees:
                start1, assigned1 = task_vars[(task["taskId"], emp1["empId"])]
                start2, assigned2 = task_vars[(dep_id, emp2["empId"])]
                model.Add(start1 >= start2 + dep_duration).OnlyEnforceIf([assigned1, assigned2])

# Objective: Prioritise high-priority tasks and minimise penalties
objective_terms = []

# Penalty: unassigned task = +100 cost
for penalty in penalties:
    objective_terms.append(penalty * 100)

# Priority weight: low = 20, medium = 10, high = 0
for task in tasks:
    priority = priority_map.get(task["priority"], 2)
    for emp in employees:
        _, assigned = task_vars[(task["taskId"], emp["empId"])]
        objective_terms.append(assigned * priority * 10)

model.Minimize(sum(objective_terms))

# Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

results = []

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    print(" Solution Found:\n")
    for task in tasks:
        task_id = task["taskId"]
        task_name = task["name"]
        duration = max(1, task.get("estimatedDurationDays", task.get("estimatedDurationHours", 8) // 8))
        assigned_flag = False
        for emp in employees:
            emp_id = emp["empId"]
            emp_name = emp["name"]
            start_day, assigned = task_vars[(task_id, emp_id)]

            if solver.Value(assigned):
                assigned_flag = True
                start = solver.Value(start_day)
                end = start + duration - 1
                print(f" Task '{task_name}' assigned to {emp_name}: Day {start} to {end}")
                results.append({
                    "task": task_name,
                    "employee": emp_name,
                    "start_day": start,
                    "end_day": end
                })

        if not assigned_flag:
            print(f"  Task '{task_name}' could not be assigned!")

    with open(r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\scheduler-ui\public\schedule_output.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n Exported to schedule_output.json")

else:
    print("No feasible solution found.")
    print("Status:", solver.StatusName(status))
