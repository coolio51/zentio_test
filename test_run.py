import numpy as np
import time

# Constants
NUM_SKILLS = 12
NUM_DAYS = 5
HOURS_PER_DAY = 24
TOTAL_TIME_SLOTS = NUM_DAYS * HOURS_PER_DAY

# Create availability vector from day/hour ranges
def create_availability(slots_per_day, time_ranges_per_day):
    availability = np.zeros(NUM_DAYS * slots_per_day, dtype=int)
    for day_index, (start, end) in enumerate(time_ranges_per_day):
        for hour in range(start, end):
            index = day_index * slots_per_day + hour
            availability[index] = 1
    return availability 

# Convert time slot index to "Wednesday at 4pm" style label
def time_slot_to_human(time_slot: int, duration: int, hours_per_day: int = 24):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_index = time_slot // hours_per_day
    hour = time_slot % hours_per_day

    if day_index >= len(days):
        return "Invalid time slot"

    day = days[day_index]
    suffix = "am" if hour < 12 else "pm"
    hour_12 = hour if hour <= 12 else hour - 12
    if hour_12 == 0:
        hour_12 = 12

    return f"{day} at {hour_12}{suffix} for {duration}h"

# Worker 1
skills_w1 = np.array([int(c) for c in "111111111010"]).reshape(NUM_SKILLS, 1)
availability_w1 = create_availability(
    HOURS_PER_DAY,
    [
        (8, 16),    # Monday
        (16, 24),   # Tuesday
        (8, 16),    # Wednesday
        (8, 16),    # Thursday
        (16, 24),   # Friday
    ]
)

# Worker 2
skills_w2 = np.array([int(c) for c in "101110111100"]).reshape(NUM_SKILLS, 1)
availability_w2 = create_availability(
    HOURS_PER_DAY,
    [
        (8, 16),     # Monday
        (8, 16),     # Tuesday
        (16, 24),    # Wednesday
        (8, 16),     # Thursday
        (16, 24),    # Friday
    ]
)

# Compute skill × time availability matrix
def compute_combined_matrix(skills: np.ndarray, availability: np.ndarray):
    start = time.perf_counter()
    availability = availability.reshape(1, -1)         # shape: (1, 120)
    combined = skills @ availability                   # shape: (12, 1) @ (1, 120) = (12, 120)
    elapsed = time.perf_counter() - start
    return combined, elapsed

# Check duration-based availability for a specific skill and start time
def check_worker_availability_for_duration(skills, availability, skill_index, start_slot, duration):
    if skills[skill_index] == 0:
        return False
    if start_slot + duration > len(availability):
        return False
    return availability[start_slot : start_slot + duration].all() == 1

# Compute matrices and times
combined_w1, time_w1 = compute_combined_matrix(skills_w1, availability_w1)
combined_w2, time_w2 = compute_combined_matrix(skills_w2, availability_w2)

# Check duration availability and time it
skill_index = 1
start_slot = 10
duration = 2
slot_human = time_slot_to_human(start_slot, duration, HOURS_PER_DAY)

start_dur_w1 = time.perf_counter()
available_w1 = check_worker_availability_for_duration(skills_w1.flatten(), availability_w1, skill_index, start_slot, duration)
time_dur_w1 = time.perf_counter() - start_dur_w1

start_dur_w2 = time.perf_counter()
available_w2 = check_worker_availability_for_duration(skills_w2.flatten(), availability_w2, skill_index, start_slot, duration)
time_dur_w2 = time.perf_counter() - start_dur_w2

# Output
print(f"Worker 1 matrix shape: {combined_w1.shape}, Computation time: {time_w1:.6f} seconds")
print(f"Worker 2 matrix shape: {combined_w2.shape}, Computation time: {time_w2:.6f} seconds")
print(f"Worker 1 available for skill {skill_index} at {slot_human}: {available_w1}, Duration check time: {time_dur_w1:.6f} seconds")
print(f"Worker 2 available for skill {skill_index} at {slot_human}: {available_w2}, Duration check time: {time_dur_w2:.6f} seconds")
