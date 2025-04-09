import json
from ortools.sat.python import cp_model
from datetime import datetime, timedelta

# Load dataset
with open(r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\Google or tools\dataset\datafiveempsixtask.json") as f:
    data = json.load(f)

employees = data['employees']
tasks = data['tasks']
rules = {r['ruleType']: r for r in data['rules'] if r['isActive']}

# Priority mapping
priority_weights = {
    "low": 1,
    "medium": 2,
    "high": 3
}

# Convert availability to a map per employee
availability_map = {}
for emp in employees:
    emp_id = emp["empId"]
    availability_map[emp_id] = {}
    for pattern in emp["availabilityPatterns"]:
        if pattern["type"] == "weekly":
            for day in pattern["days"]:
                start = datetime.strptime(pattern["startTime"], "%H:%M")
                end = datetime.strptime(pattern["endTime"], "%H:%M")
                availability_map[emp_id][day] = (start, end)

# Skill map
def has_required_skills(emp, required_skills):
    emp_skills = {s["skillName"]: s["level"] for s in emp["skills"]}
    for r in required_skills:
        if r["skillName"] not in emp_skills or emp_skills[r["skillName"]] < r.get("minLevel", 1):
            return False
    return True

# Time constants
DAY_START = 9
DAY_END = 17
TOTAL_DAYS = 5
MAX_HOURS = (DAY_END - DAY_START) * TOTAL_DAYS

model = cp_model.CpModel()

assignments = {}
starts = {}
ends = {}
all_tasks = range(len(tasks))
all_emps = range(len(employees))

# Task durations and priorities
task_durations = [t["estimatedDurationHours"] for t in tasks]
task_priorities = [priority_weights[t["priority"]] for t in tasks]

# Assignment vars
for t_id, task in enumerate(tasks):
    for e_id, emp in enumerate(employees):
        if not has_required_skills(emp, task["requiredSkills"]):
            continue
        for day in range(TOTAL_DAYS):
            start_range = list(range(DAY_START, DAY_END - task_durations[t_id] + 1))
            for hour in start_range:
                key = (t_id, e_id, day, hour)
                var = model.NewBoolVar(f"assign_t{t_id}_e{e_id}_d{day}_h{hour}")
                assignments[key] = var

# Each task must be assigned once
for t_id in all_tasks:
    possible_assignments = []
    for key in assignments:
        if key[0] == t_id:
            possible_assignments.append(assignments[key])
    model.AddExactlyOne(possible_assignments)

# No employee can have overlapping tasks
for e_id in all_emps:
    for day in range(TOTAL_DAYS):
        for hour in range(DAY_START, DAY_END):
            overlapping = []
            for t_id in all_tasks:
                dur = task_durations[t_id]
                for h_start in range(max(DAY_START, hour - dur + 1), min(hour + 1, DAY_END - dur + 1)):
                    key = (t_id, e_id, day, h_start)
                    if key in assignments:
                        overlapping.append(assignments[key])
            if overlapping:
                model.Add(sum(overlapping) <= 1)

# Respect availability
for key in list(assignments.keys()):
    t_id, e_id, day, hour = key
    emp = employees[e_id]
    emp_id = emp["empId"]
    if day not in availability_map[emp_id]:
        model.Add(assignments[key] == 0)
        continue
    start, end = availability_map[emp_id][day]
    task_end = datetime.strptime(f"{hour + task_durations[t_id]}:00", "%H:%M")
    if not (start.time() <= datetime.strptime(f"{hour}:00", "%H:%M").time() < end.time() and
            task_end.time() <= end.time()):
        model.Add(assignments[key] == 0)

# Dependencies
task_id_map = {task["taskId"]: i for i, task in enumerate(tasks)}
for t_id, task in enumerate(tasks):
    for dep in task.get("dependencies", []):
        dep_id = task_id_map[dep]
        for k1 in assignments:
            if k1[0] != t_id:
                continue
            for k2 in assignments:
                if k2[0] != dep_id:
                    continue
                # Enforce that dep ends before current starts
                model.Add(k1[2] * 24 + k1[3] >= k2[2] * 24 + k2[3] + task_durations[dep_id]).OnlyEnforceIf([assignments[k1], assignments[k2]])

# Objective: maximise weighted priority tasks
model.Maximize(sum(assignments[k] * task_priorities[k[0]] for k in assignments))

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for k in assignments:
        if solver.Value(assignments[k]):
            t_id, e_id, day, hour = k
            start = f"{hour:02d}:00"
            end = f"{hour + task_durations[t_id]:02d}:00"
            print(f"Assigned {tasks[t_id]['name']} to {employees[e_id]['name']} on day {day} from {start} to {end}")
else:
    print("No feasible schedule found.")
