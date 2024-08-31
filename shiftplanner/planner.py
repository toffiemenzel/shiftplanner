from flask import request
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
import csv, json

# Constants
day_indices = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
genders = ['m', 'w', 'd']

def run_offline(file_path):
    # Open and load the JSON file
    with open(file_path, 'r') as state_file:
        app_data = json.load(state_file)
    
    # Call the generate_schedule function with the loaded app_data
    return generate_schedule(app_data)
    

def generate_schedule(app_data):
    # Shift Parameters
    shift_duration_hours = int(app_data.get('shift_params', {}).get('shift_duration_hours', 0))  # Duration of each shift in hours
    total_shifts = int(app_data.get('shift_params', {}).get('num_shifts', 0)) # Number of shifts

    # Planner Parameters
    num_assignments_per_person = int(app_data.get('num_assignments_per_person', 0))
    opt_consider_travel = app_data.get('opt_consider_travel', True)
    opt_balance_gender = app_data.get('opt_balance_gender', True)
    opt_same_time_slots = app_data.get('opt_same_time_slots', True)
    opt_max_shift_dist = app_data.get('opt_max_shift_dist', True)
    opt_enforce_shift_dist = app_data.get('opt_enforce_shift_dist', True)
    min_distance_between_shifts = int(app_data.get('min_distance_between_shifts', 2))+1
    opt_match_partners = app_data.get('opt_match_partners', True)
    partner_bonus = int(app_data.get('partner_bonus', 10))  # Bonus for each pair of preferred partners working the same shift
    experience_penalty = int(app_data.get('experience_penalty', 100))  # Penalty for imbalance in experience for shifts for relevant roles
    penalty_outside_window = int(app_data.get('penalty_outside_window', 1000))  # Penalty for assigning a shift outside of a person's availability
    gender_penalty = int(app_data.get('gender_penalty', 10))  # Penalty for gender imbalance in a shift
    penalty_for_same_time_slot = int(app_data.get('penalty_for_same_time_slot', 30))
    extra_shifts_key = "extra_shifts"
    solver_timeout_sec = int(app_data.get('solver_timeout_sec', 30))

    # Data
    persons = app_data.get('persons',{})
    roles = app_data.get('roles',{})
    role_experience_required = app_data.get('role_experience_required',{})
    shift_names = app_data.get('shift_names',{})
    shift_table = app_data.get('shift_table',{})
    shift_datetimes = app_data.get('shift_datetimes',{})

    ######################################################################################################################

    # Calculate the required and available person-shifts
    required_person_shifts = sum(sum(role) for role in shift_table)
    available_person_shifts = len(persons) * num_assignments_per_person
    plus_shifts = available_person_shifts - required_person_shifts

    # If there are more available shifts than required, adjust the shift_table
    if plus_shifts > 0:
        # Identify roles (and corresponding indices) with multiple values
        expandable_roles = {role: shift_table[roles.index(role)] for role in roles if len(set(shift_table[roles.index(role)])) > 1}
        print(f"Expandable roles: {expandable_roles}")
        if expandable_roles:
            for _ in range(plus_shifts):
                # Choose the role to expand by finding the role with multiple values
                for role, counts in expandable_roles.items():
                    print(f"Role: {role} / count: {counts}")
                    
                    i = [c for c in counts if c < max(counts)]
                    if i:
                        second_max_value = max(i)
                        # Find the last occurrence of the second-highest value and increase it by 1
                        last_index = len(counts) - 1 - counts[::-1].index(second_max_value)
                    else:
                        i = int(0.66*(len(counts) - 1))

                    shift_table[roles.index(role)][last_index] += 1

                    # Break out of the loop to reassess plus_shifts after every increment
                    break
        else:
            # If no roles have multiple values, find the role with the highest single value and start individualizing it per shift
            max_role_index = max(range(len(roles)), key=lambda i: shift_table[i][0])
            
            # Calculate the 3/4 position (rounded down)
            target_index = int(3 * (total_shifts - 1) / 4)
            shift_table[max_role_index][target_index] += 1
            plus_shifts -= 1

            # Adjust the remaining plus_shifts by applying the rule of the last position of the second highest element
            counts = shift_table[max_role_index]
            for _ in range(plus_shifts):
                max_value = max(counts)
                if target_index > 0:
                    target_index -= 1
                counts[target_index] += 1

    # Derived parameters
    num_people = len(persons)
    num_shifts = len(shift_names)

    # Validate roles in people_data against the valid roles list
    for person in persons:
        for role in person['experienced_roles'] + person['inexperienced_roles']:
            if role not in roles:
                raise ValueError(f"Error: Invalid role '{role}' found for {person['name']}")

    # Determine if we have enough people to run the show
    enough_shifts = available_person_shifts >= required_person_shifts
    extra_shifts = any(extra_shifts_key in person["options"] for person in persons)

    print("Required shifts:", required_person_shifts, "Available shifts:", available_person_shifts)
    if not enough_shifts and not extra_shifts:
        print()
        result = {
                'status': f"Error: Not enough people to fill all shifts. Required shifts: {required_person_shifts}, Available shifts: {available_person_shifts}",
                'stats': {},
                'people_table': [],
                'shift_table': []
            }
        return result
    else:
        # Create the model
        model = cp_model.CpModel()
        conditions = []

        # Create time slots for constraints on shifts
        time_slots = [name.split()[1].split('-')[0] for name in shift_names]

        # Convert shift_datetimes from strings to datetime objects if necessary
        shift_datetimes = [
            datetime.fromisoformat(dt) if isinstance(dt, str) else dt
            for dt in app_data.get('shift_datetimes', [])
        ]

        # Ensure all times are timezone-naive before comparison
        shift_datetimes = [dt.replace(tzinfo=None) for dt in shift_datetimes]

        # Convert arrival and departure times to indices in people_data
        for person in persons:          
            arrival_day, arrival_hour = person['arrival_time'].replace('!', '').split() if person['arrival_time'] else (None, None)
            departure_day, departure_hour = person['departure_time'].replace('!', '').split() if person['departure_time'] else (None, None)

            if arrival_day and arrival_hour:
                arrival_time = datetime(2024, 1, 1) + timedelta(days=day_indices[arrival_day], hours=int(arrival_hour))
                earliest_shift = next(idx for idx, dt in enumerate(shift_datetimes) if dt.replace(tzinfo=None) >= arrival_time.replace(tzinfo=None))
            else:
                earliest_shift = 0  # No constraint on earliest shift if arrival_time is empty
            
            if departure_day and departure_hour:
                departure_time = datetime(2024, 1, 1) + timedelta(days=day_indices[departure_day], hours=int(departure_hour))
                latest_shift = next((idx for idx, dt in enumerate(shift_datetimes) 
                            if dt.replace(tzinfo=None) + timedelta(hours=shift_duration_hours) > departure_time.replace(tzinfo=None)), total_shifts) - 1
            else:
                latest_shift = total_shifts - 1  # No constraint on latest shift if departure_time is empty
            
            person['earliest_shift'] = earliest_shift
            person['latest_shift'] = latest_shift

        # Extract the pre-assigned and vetoed shifts
        for person in persons:    
            person['assign_shifts_parsed'] = []
            person['veto_shifts_parsed'] = []

            if 'assign_shifts' in person and person['assign_shifts']:
                assign_shifts_list = person['assign_shifts'].split(',')
                for assign in assign_shifts_list:
                    role, shift_name = assign.strip().split('(')
                    shift_name = shift_name.rstrip(')')
                    person['assign_shifts_parsed'].append((role.strip(), shift_name.strip()))

            if 'veto_shifts' in person and person['veto_shifts']:
                person['veto_shifts_parsed'] = [shift.strip() for shift in person['veto_shifts'].split(',')]

        ##########################################################################################

        # Decision variables
        shifts = {}
        penalties = []  # List to track penalties for scheduling outside of the availability window
        for i, person in enumerate(persons):
            for j in range(num_shifts):
                for r in roles:
                    if r in person['experienced_roles'] + person['inexperienced_roles']:
                        shift_var = model.NewBoolVar(f'shift_{i}_{j}_{r}')
                        shifts[(i, j, r)] = shift_var
                        
                        # Add penalties for assigning shifts outside availability window
                        if opt_consider_travel and ((j < person['earliest_shift'] or j > person['latest_shift']) and not person['arrival_hard'] and not person['departure_hard']):
                            penalties.append(shift_var * penalty_outside_window)
                            conditions.append(f'shift_{person["name"]}_{j}_{r} == 1 increases the penalty')
                        
                        # Enforce hard constraints (no shifts can be assigned outside the hard limits)
                        if (j < person['earliest_shift'] and person['arrival_hard']) or (j > person['latest_shift'] and person['departure_hard']):
                            shifts[(i, j, r)] = model.NewConstant(0)
                            conditions.append(f'shift_{person["name"]}_{j}_{r} == 0')
                    else:
                        shifts[(i, j, r)] = model.NewConstant(0)
                        conditions.append(f'shift_{person["name"]}_{j}_{r} == 0')
        
        # Soft constraint: Try to balance experienced and inexperienced persons for roles requiring experience mix
        imbalance_vars = [] 
        for j in range(num_shifts):
            for r in roles:
                if role_experience_required[r]:
                    experienced = sum(shifts[(i, j, r)] for i in range(num_people) if r in persons[i]['experienced_roles'])
                    inexperienced = sum(shifts[(i, j, r)] for i in range(num_people) if r in persons[i]['inexperienced_roles'])
                    imbalance = model.NewIntVar(0, num_shifts, f'imbalance_{j}_{r}')
                    model.AddAbsEquality(imbalance, experienced - inexperienced)
                    conditions.append(f'imbalance_{j}_{r} == abs({experienced} - {inexperienced})')
                    penalties.append(imbalance * experience_penalty)
                    conditions.append(f'imbalance_{j}_{r} * {experience_penalty} is added to the penalty')
                    imbalance_vars.append((j, r, imbalance))
        
        # Soft constraint: Penalize large disparities in gender counts within each shift
        gender_diff_vars = {}
        if opt_balance_gender:
            for j in range(num_shifts):
                gender_counts = {gender: model.NewIntVar(0, num_people, f'gender_{gender}_{j}') for gender in genders}
                
                # Count the number of people of each gender in the shift
                for gender in gender_counts:
                    model.Add(gender_counts[gender] == sum(shifts[(i, j, r)] for i in range(num_people) for r in roles if persons[i]['gender'] == gender))
                    conditions.append(f'gender_{gender}_{j} == sum of {gender} counts in shift {j}')
                
                # Penalize the absolute differences between the gender counts
                gender_diffs = []
                gender_diff_vars[j] = []
                gender_list = list(gender_counts.keys())
                for i in range(len(gender_list)):
                    for k in range(i + 1, len(gender_list)):
                        diff = model.NewIntVar(0, num_people, f'diff_{gender_list[i]}_{gender_list[k]}_{j}')
                        model.AddAbsEquality(diff, gender_counts[gender_list[i]] - gender_counts[gender_list[k]])
                        conditions.append(f'diff_{gender_list[i]}_{gender_list[k]}_{j} == abs(gender_{gender_list[i]}_{j} - gender_{gender_list[k]}_{j})')

                        gender_diffs.append(diff)
                        gender_diff_vars[j].append(diff)
                
                penalties.extend(gender_diffs)
                conditions.append(f'Penalty for gender disparities in shift {j}: {gender_diffs}')
        
        # Each person gets exactly n shifts
        for i in range(num_people):
            if extra_shifts_key in persons[i]['options'] and not enough_shifts:
                model.Add(sum(shifts[(i, j, r)] for j in range(num_shifts) for r in persons[i]['experienced_roles'] + persons[i]['inexperienced_roles']) >= num_assignments_per_person)   
                conditions.append(f'{persons[i]["name"]} is allowed at least {num_assignments_per_person} shifts due to extra shifts option')
            else:
                model.Add(sum(shifts[(i, j, r)] for j in range(num_shifts) for r in persons[i]['experienced_roles'] + persons[i]['inexperienced_roles']) == num_assignments_per_person)  
                conditions.append(f'{persons[i]["name"]} must be assigned exactly {num_assignments_per_person} shifts')  

        # Each person can have at most one role per shift
        for i in range(num_people):
            for j in range(num_shifts):
                model.Add(sum(shifts[(i, j, r)] for r in roles) <= 1)
                conditions.append(f'{persons[i]["name"]} can have at most one role in shift {j}')

        # Each shift must have exactly the required number of each role
        for j in range(num_shifts):
            for r in roles:
                role_index = roles.index(r)
                model.Add(sum(shifts[(i, j, r)] for i in range(num_people)) == shift_table[role_index][j])
                conditions.append(f'Shift {j} must have exactly {shift_table[role_index][j]} people assigned to role {r}')
        
        # Enforce pre-assigned shifts
        for i, person in enumerate(persons):
            if 'assign_shifts_parsed' in person:
                for role, shift_name in person['assign_shifts_parsed']:
                    # Check if the shift_name exists in the shift_names list
                    if shift_name not in shift_names:
                        print(f"Warning: Shift name '{shift_name}' for {person['name']} is not valid and will be ignored.")
                        continue

                    # Get the index of the shift
                    j = shift_names.index(shift_name)

                    if role:
                        # If role is specified, check if it exists in the roles list
                        if role not in roles:
                            print(f"Warning: Role '{role}' for {person['name']} is not valid and will be ignored.")
                            continue
                        
                        # Add a hard constraint that the person must be assigned to the specific role in the specific shift
                        print(f"{role} / {shift_name} -> j = {j}")
                        shift_var = shifts[(i, j, role)]
                        model.Add(shift_var == 1)
                        conditions.append(f"{person['name']} must be assigned to role {role} in shift {shift_name}")
                    else:
                        # If role is empty, person must be assigned to any of their available roles for this shift
                        available_roles = person['experienced_roles'] + person['inexperienced_roles']
                        valid_roles = [r for r in available_roles if r in roles]
                        
                        if valid_roles:
                            # Create a condition that ensures assignment to any valid role in the given shift
                            model.Add(sum(shifts[(i, j, r)] for r in valid_roles) == 1)
                            conditions.append(f"{person['name']} must be assigned to any available role in shift {shift_name}")
                        else:
                            print(f"Warning: {person['name']} has no valid roles for shift '{shift_name}'.")

        # Enforce vetos for shifts
        for i, person in enumerate(persons):
            if 'veto_shifts_parsed' in person:
                for shift_name in person['veto_shifts_parsed']:
                    # Check if the shift_name exists in the shift_names list
                    if shift_name not in shift_names:
                        print(f"Warning: Shift name '{shift_name}' in veto list for {person['name']} is not valid and will be ignored.")
                        continue

                    # If the check passes, proceed to enforce the veto
                    j = shift_names.index(shift_name)
                    for r in roles:
                        shift_var = shifts[(i, j, r)]
                        model.Add(shift_var == 0)
                        conditions.append(f"{person['name']} cannot be assigned to any role in shift {shift_name}")

        # Enforce minimum distance between shifts for each person
        if opt_enforce_shift_dist:
            for i in range(num_people):
                for j1 in range(num_shifts):
                    for j2 in range(j1 + 1, num_shifts):
                        if abs(j2 - j1) < min_distance_between_shifts:
                            for r1 in persons[i]['experienced_roles'] + persons[i]['inexperienced_roles']:
                                for r2 in persons[i]['experienced_roles'] + persons[i]['inexperienced_roles']:
                                    model.Add(shifts[(i, j1, r1)] + shifts[(i, j2, r2)] <= 1)
                                    conditions.append(f'{persons[i]["name"]} cannot work in {r1} during shift {j1} and {r2} during shift {j2} because the shifts are less than {min_distance_between_shifts} hours apart')
        
        # Soft constraint: Penalize assigning more than one shift per time slot type per person
        if opt_same_time_slots:
            time_slot_penalties = []  # List to hold penalties related to time slots
            for i in range(num_people):
                for time_slot in set(time_slots):
                    # Create a variable representing the number of shifts in this time slot
                    shifts_in_slot = model.NewIntVar(0, num_shifts, f'shifts_in_slot_{i}_{time_slot}')

                    # Sum the shifts in the given time slot for this person
                    model.Add(shifts_in_slot == sum(shifts[(i, j, r)] for j in range(num_shifts) if time_slots[j] == time_slot for r in persons[i]['experienced_roles'] + persons[i]['inexperienced_roles']))
                    conditions.append(f'shifts_in_slot_{persons[i]["name"]}_{time_slot} == sum of shifts in slot {time_slot}')

                    # Create a variable for the penalty if more than one shift is assigned
                    over_assignment = model.NewIntVar(0, num_shifts, f'over_assignment_{i}_{time_slot}')
                    
                    # Instead of max(), use this constraint to handle the penalty calculation
                    model.Add(over_assignment >= shifts_in_slot - 1)
                    model.Add(over_assignment <= shifts_in_slot)

                    # Add the penalty to the penalties list
                    penalty = over_assignment * penalty_for_same_time_slot
                    time_slot_penalties.append(penalty)
                    
                    conditions.append(f'over_assignment_{persons[i]["name"]}_{time_slot} > 0 increases the penalty')

            # Add the penalties to the overall penalties list
            penalties.extend(time_slot_penalties)

        # Maximize the distance between assigned shifts
        distances = []
        if opt_max_shift_dist:
            for i in range(num_people):
                for j1 in range(num_shifts):
                    for j2 in range(j1 + 1, num_shifts):
                        if abs(j2 - j1) >= min_distance_between_shifts:
                            for r1 in persons[i]['experienced_roles'] + persons[i]['inexperienced_roles']:
                                for r2 in persons[i]['experienced_roles'] + persons[i]['inexperienced_roles']:
                                    distance = model.NewIntVar(0, num_shifts, f'distance_{i}_{j1}_{j2}_{r1}_{r2}')
                                    model.AddMultiplicationEquality(distance, [(j2 - j1), shifts[(i, j1, r1)], shifts[(i, j2, r2)]])
                                    distances.append(distance)
                                    conditions.append(f'Distance for {persons[i]["name"]} between shift {j1} ({r1}) and shift {j2} ({r2}) is maximized with distance variable {distance}.')

        # Soft constraint: Try to schedule preferred partners together
        total_bonus = []
        if opt_match_partners:
            for i, person in enumerate(persons):
                preferred_partners_list = [partner.strip() for partner in person['preferred_partners'].split(',')]
                for partner_name in preferred_partners_list:
                    partner_index = next((index for index, p in enumerate(persons) if p['name'] == partner_name), None)
                    if partner_index is not None:
                        for j in range(num_shifts):
                            # Boolean variables representing whether each person is assigned to this shift
                            assigned_to_shift_i = model.NewBoolVar(f'assigned_to_shift_i_{i}_{j}')
                            assigned_to_shift_partner = model.NewBoolVar(f'assigned_to_shift_partner_{partner_index}_{j}')
                            
                            model.AddMaxEquality(assigned_to_shift_i, [shifts[(i, j, r1)] for r1 in persons[i]['experienced_roles'] + persons[i]['inexperienced_roles']])
                            model.AddMaxEquality(assigned_to_shift_partner, [shifts[(partner_index, j, r2)] for r2 in persons[partner_index]['experienced_roles'] + persons[partner_index]['inexperienced_roles']])

                            conditions.append(f'{person["name"]} (shift {j}) assigned to shift {j} if any of their roles are assigned (assigned_to_shift_{person["name"]}_{j} == 1)')
                            conditions.append(f'{partner_name} (shift {j}) assigned to shift {j} if any of their roles are assigned (assigned_to_shift_partner_{partner_name}_{j} == 1)')
                            
                            # Reward if both are assigned to the same shift
                            same_shift = model.NewBoolVar(f'same_shift_{i}_{partner_index}_{j}')
                            model.AddBoolAnd([assigned_to_shift_i, assigned_to_shift_partner]).OnlyEnforceIf(same_shift)
                            model.AddBoolOr([assigned_to_shift_i.Not(), assigned_to_shift_partner.Not()]).OnlyEnforceIf(same_shift.Not())

                            conditions.append(f'Represent: same_shift_{person["name"]}_{partner_name}_{j}')
                            
                            total_bonus.append(same_shift * partner_bonus)
                            conditions.append(f'same_shift_{person["name"]}_{partner_name}_{j} == 1 will increase the bonus')

        # Maximize the total distance, bonuses, and minimize penalties
        model.Maximize(sum(total_bonus) + sum(distances) - sum(penalties))

        # Print the conditions for debugging
        '''
        with open("conditions_output.txt", 'w') as f:
            for condition in conditions:
                f.write(condition + '\n')
        '''    
        
        # Create the solver and solve
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8 
        solver.parameters.max_time_in_seconds = solver_timeout_sec
        status = solver.Solve(model)

        # After solving the model, calculate and print statistics
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            '''
            # Save the first table (people table) to CSV
            with open('people.csv', 'w', newline='') as csvfile:
                people_writer = csv.writer(csvfile)
                # Write header
                people_writer.writerow(['Person'] + shift_names)
                
                for i, person in enumerate(persons):
                    row = [person['name']]
                    for j in range(num_shifts):
                        shifts_str = ''.join([r if solver.BooleanValue(shifts[(i, j, r)]) else '' for r in roles])
                        row.append(shifts_str)
                    
                    # Write row to CSV
                    people_writer.writerow(row)
            
            # Create and save the second table (role-assignment table) to CSV
            role_assignments = {r: [""] * num_shifts for r in roles}
            
            for j in range(num_shifts):
                for r in roles:
                    assigned_people = [persons[i]['name'] for i in range(num_people) if solver.BooleanValue(shifts[(i, j, r)])]
                    role_assignments[r][j] = ", ".join(assigned_people) if assigned_people else ""
            
            with open('shifts.csv', 'w', newline='') as csvfile:
                shifts_writer = csv.writer(csvfile)
                # Write header
                shifts_writer.writerow(['Role'] + shift_names)
                
                # Write rows for each role
                for r in roles:
                    row = [r] + role_assignments[r]
                    shifts_writer.writerow(row)

            # Print tables to the console
            print(f"{'Person':<11} | " + " | ".join([f"{name:^11}" for name in shift_names]) + " |")
            print("-" * (10 + 2 + (num_shifts * 14)) + "|")
            
            for i, person in enumerate(persons):
                assigned_shifts = [(j, r) for j in range(num_shifts) for r in roles if solver.BooleanValue(shifts[(i, j, r)])]
                
                print(f"{person['name']:<11} | " +
                    " | ".join([f"{''.join([r if solver.BooleanValue(shifts[(i, j, r)]) else '' for r in roles]):<11}" for j in range(num_shifts)]) +
                    " | ")
            
            # Printing the role-assignment table
            print("\nRole Assignment Table:")
            print(f"{'Role':<11} | " + " | ".join([f"{name:^11}" for name in shift_names]) + " |")
            print("-" * (10 + 3 + (num_shifts * 12)) + " |")
            
            for r in roles:
                print(f"{r:<11} | " + " | ".join([f"{names:<11}" for names in role_assignments[r]]) + " |")
            '''
            people_table, shift_table = generate_shift_tables(solver, shifts, persons, roles, shift_names, num_shifts)
            max_shift_distance_score = calculate_max_shift_distance_score(solver, shifts, persons, roles, num_shifts)
            gender_parity_score = calculate_gender_parity_score(solver, shifts, persons, roles, num_shifts)
            time_slot_repetition_score = calculate_time_slot_repetition_score(solver, shifts, persons, roles, num_shifts, time_slots)
            time_window_violation_score = calculate_time_window_violation_score(solver, shifts, persons, roles, num_shifts)
            experience_balance_score = calculate_experience_balance_score(solver, shifts, persons, roles, role_experience_required, num_shifts)
            partner_matching_score = calculate_partner_matching_score(solver, shifts, persons, num_shifts, roles)
            stats = {
                'score': solver.ObjectiveValue(),
                'max_shift_distance_score': max_shift_distance_score,
                'gender_parity_score': gender_parity_score,
                'time_slot_repetition_score': time_slot_repetition_score,
                'time_window_violation_score': time_window_violation_score,
                'experience_balance_score': experience_balance_score,
                'partner_matching_score': partner_matching_score
            }
            print(stats)
            result = {
                'status': "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                'stats': stats,
                'people_table': people_table,
                'shift_table': shift_table
            }
        else:
            result = {
                'status': "NOT FEASIBLE",
                'stats': {},
                'people_table': [],
                'shift_table': []
            }
        print("PLANNER RESULT: " + result.get('status', "UNKNOWN"))
        return result

def generate_shift_tables(solver, shifts, persons, roles, shift_names, num_shifts):
    # Initialize the 2D lists for the tables
    people_table = []
    role_assignment_table = []
    
    # Create the header row for the people table
    people_header = ['Person'] + shift_names
    people_table.append(people_header)
    
    # Fill the people table with the shift assignments for each person
    for i, person in enumerate(persons):
        row = [person['name']]
        for j in range(num_shifts):
            shifts_str = ''.join([r if solver.BooleanValue(shifts[(i, j, r)]) else '' for r in roles])
            row.append(shifts_str)
        people_table.append(row)
    
    # Create the header row for the role-assignment table
    role_header = ['Role'] + shift_names
    role_assignment_table.append(role_header)
    
    # Fill the role-assignment table with the assignments of people to each role in each shift
    role_assignments = {r: [""] * num_shifts for r in roles}
    
    for j in range(num_shifts):
        for r in roles:
            assigned_people = [persons[i]['name'] for i in range(len(persons)) if solver.BooleanValue(shifts[(i, j, r)])]
            role_assignments[r][j] = ", ".join(assigned_people) if assigned_people else ""
    
    for r in roles:
        row = [r] + role_assignments[r]
        role_assignment_table.append(row)
    
    return people_table, role_assignment_table


# Function to calculate the maximum distance score
def calculate_max_shift_distance_score(solver, shifts, persons, roles, num_shifts):
    total_score = 0
    max_possible_distance = num_shifts - 1  # Distance between the first and last shifts

    for i, person in enumerate(persons):
        assigned_shifts = []
        for j in range(num_shifts):
            if any(solver.BooleanValue(shifts[(i, j, r)]) for r in roles):
                assigned_shifts.append(j)

        if len(assigned_shifts) > 1:
            if len(assigned_shifts) > 2:
                # If a person has more than two shifts, take the minimum distance between their shifts
                min_distance = min(assigned_shifts[k+1] - assigned_shifts[k] for k in range(len(assigned_shifts) - 1))
            else:
                # If a person has exactly two shifts, the distance is the difference between these two shifts
                min_distance = assigned_shifts[-1] - assigned_shifts[0]

            # Normalize the score for this person
            person_score = (min_distance / max_possible_distance) * 100
            total_score += person_score

    # Average score over all persons
    average_score = total_score / len(persons)
    return average_score

# Function to calculate the gender parity score
def calculate_gender_parity_score(solver, shifts, persons, roles, num_shifts):
    total_score = 0

    for j in range(num_shifts):
        male_count = 0
        female_count = 0
        diverse_count = 0

        for i, person in enumerate(persons):
            if any(solver.BooleanValue(shifts[(i, j, r)]) for r in roles):
                if person['gender'] == 'm':
                    male_count += 1
                elif person['gender'] == 'w':
                    female_count += 1
                elif person['gender'] == 'd':
                    diverse_count += 1

        # Determine the larger and smaller groups
        larger_group = max(male_count, female_count)
        smaller_group = min(male_count, female_count)

        # Calculate the score for this shift
        if smaller_group + diverse_count >= larger_group:
            shift_score = 0
        else:
            shift_score = larger_group - (smaller_group + diverse_count)

        total_score += shift_score

    return total_score

# Function to calculate the time slot repetition score
def calculate_time_slot_repetition_score(solver, shifts, persons, roles, num_shifts, time_slots):
    total_score = 0

    for i, person in enumerate(persons):
        # Dictionary to track how many times each time slot occurs
        time_slot_counts = {}

        for j in range(num_shifts):
            if any(solver.BooleanValue(shifts[(i, j, r)]) for r in roles):
                time_slot = time_slots[j]
                if time_slot in time_slot_counts:
                    time_slot_counts[time_slot] += 1
                else:
                    time_slot_counts[time_slot] = 1

        # Calculate the score for this person
        for count in time_slot_counts.values():
            if count > 1:
                total_score += (count - 1)

    return total_score

# Function to calculate the time window constraint violation score
def calculate_time_window_violation_score(solver, shifts, persons, roles, num_shifts):
    total_score = 0

    for i, person in enumerate(persons):
        earliest_shift = person['earliest_shift']
        latest_shift = person['latest_shift']
        for j in range(num_shifts):
            if j < earliest_shift or j > latest_shift:
                if any(solver.BooleanValue(shifts[(i, j, r)]) for r in roles):
                    total_score += 1

    return total_score

# Function to calculate the experience balance score
def calculate_experience_balance_score(solver, shifts, persons, roles, role_experience_required, num_shifts):
    total_score = 0

    for j in range(num_shifts):
        for r in roles:
            if role_experience_required.get(r, False):
                experienced_count = sum(solver.BooleanValue(shifts[(i, j, r)]) for i in range(len(persons)) if r in persons[i]['experienced_roles'])
                inexperienced_count = sum(solver.BooleanValue(shifts[(i, j, r)]) for i in range(len(persons)) if r in persons[i]['inexperienced_roles'])
                
                # Calculate the absolute difference between experienced and inexperienced counts
                balance_difference = abs(experienced_count - inexperienced_count)
                total_score += balance_difference

    return total_score

# Function to calculate the partner matching bonus score
def calculate_partner_matching_score(solver, shifts, persons, num_shifts, roles):
    total_score = 0

    for i, person in enumerate(persons):
        preferred_partners = person.get('preferred_partners', "")

        if not preferred_partners:
            continue  # Skip if no preferred partners

        # Split the preferred partners string into a list
        preferred_partners_list = [partner.strip() for partner in preferred_partners.split(',')]

        for j in range(num_shifts):
            # Check if the person is assigned to this shift
            person_assigned = any(solver.BooleanValue(shifts[(i, j, r)]) for r in roles)
            if person_assigned:
                # Count how many of their preferred partners are NOT in the same shift
                missing_partners_count = 0
                for partner_name in preferred_partners_list:
                    partner_index = next((index for index, p in enumerate(persons) if p['name'] == partner_name), None)
                    if partner_index is not None:
                        partner_assigned = any(solver.BooleanValue(shifts[(partner_index, j, r)]) for r in roles)
                        if not partner_assigned:
                            missing_partners_count += 1

                # Add to the total score
                total_score += missing_partners_count

    return total_score

