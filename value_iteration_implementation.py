import random

# Initialize an empty table
initial_table = [['-' for _ in range(15)] for _ in range(10)]

# Define the coordinates to be updated to "E"
end_coordinates = [(5, 6), (6, 6), (7, 6), (8, 6), (5, 5), (6, 5), (7, 5), (8, 5)]

# Define wall coordinates
wall_coordinates = [
    (x, y) for x in range(5, 10) for y in range(5)
] + [
    (0, y) for y in range(15)
] + [
    (9, y) for y in range(15)
] + [
    (x, 0) for x in range(10)
] + [
    (x, 14) for x in range(10)
] + [
    (4, y) for y in range(4, 11)
] + [
    (5, 10)
]

# Define start coordinates
start_coordinates = [(1, 1), (2, 1), (3, 1), (4, 1)]

# Update specific cells to "E"
for x, y in end_coordinates:
    initial_table[x][y] = 'E'

# Update cells according to wall coordinates
for x, y in wall_coordinates:
    initial_table[x][y] = 'W'

# Update cells according to start coordinates
for x, y in start_coordinates:
    initial_table[x][y] = 'S'

# Generate all possible combinations
all_combinations = [(x, y, vx, vy)
              for x in range(10)
              for y in range(15)
              for vx in range(-2, 3)
              for vy in range(-2, 3)]


# Definition of the values and policy structure
values = {}
policy = {}

# Initialize the value for each state
for state in all_combinations:
    x, y, vx, vy = state

    # Check if the car has hit a wall
    if (x, y) in wall_coordinates:
        values[state] = -10  # Penalty for hitting a wall
    # Check if the car has finished the race
    elif (x, y) in end_coordinates:
        values[state] = 100  # Reward for reaching the finishing cells
    else:
        values[state] = 0  # Default cost for the rest of the states is set to 0

# Function that checks if a path from (x1, y1) to (x2, y2) hits a wall
def hits_wall(x1, y1, x2, y2):
    if (x2, y2) in wall_coordinates:
        return True
    if (x1 == x2 and y1 == y2):
        return False

    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))

    for step in range(1, steps + 1):
        intermediate_x = x1 + step * dx // steps
        intermediate_y = y1 + step * dy // steps
        if (intermediate_x, intermediate_y) in wall_coordinates:
            return True

    return False

# Function that calculates the reward of the new state, based on the action made
def calculate_reward(current_state, new_state):
    x, y, vx, vy = current_state
    new_x, new_y, new_vx, new_vy = new_state

    if (new_x, new_y) in end_coordinates:
      return 100  # Positive reward for reaching the finishing cells
    elif (new_x, new_y) in wall_coordinates:
      return -10  # Penalty for hitting a wall
    elif (hits_wall(x, y, new_x, new_y)):
      return -10
    else:
      return -1  # Penalty for non-terminal states


# Definition of convergence threshold
CONVERGENCE_THRESHOLD = 0.000001

# Value Iteration Loop
converged = False
while not converged:
    delta = 0
    new_values = values.copy()  # Use a copy to update values

    # Iterate over all states in the environment
    for state in all_combinations:
        x, y, vx, vy = state
        old_value = values[state]

        if (x, y) not in wall_coordinates and (x, y) not in end_coordinates:
            max_state_value = float('-inf')
            best_action = (0, 0)  # Default action if none is found

            # Iterate over all possible velocity changes
            for delta_vx in range(-1, 2):
                for delta_vy in range(-1, 2):
                    new_vx = vx + delta_vx
                    new_vy = vy + delta_vy

                    # Check if the new velocity is within bounds
                    if -2 <= new_vx <= 2 and -2 <= new_vy <= 2:
                        new_x = x + new_vx
                        new_y = y + new_vy

                        # Check if the new position is within the grid boundaries
                        if 0 <= new_x < 10 and 0 <= new_y < 15:
                            new_state = (new_x, new_y, new_vx, new_vy)
                            reward = calculate_reward(state, new_state)

                            # Penalize if hitting a wall
                            if hits_wall(x, y, new_x, new_y):
                                reward = -10

                            # Update max_state_value with the immediate reward and the value of the new state
                            state_value = reward + values.get(new_state, 0)
                            if state_value > max_state_value:
                                max_state_value = state_value
                                best_action = (delta_vx, delta_vy)

            new_values[state] = max_state_value
            policy[state] = best_action  # Store the best action for this state
            delta = max(delta, abs(old_value - new_values[state]))

    values = new_values

    if delta < CONVERGENCE_THRESHOLD:
        converged = True

# Printing the values and policy for verification
#for state in sorted(values.keys()):
#  print(f"State: {state}, Value: {values[state]}, Policy: {policy.get(state)}")

# The policy dictionary now contains the best action for each state

# Function to calculate the cost of a route and print the route
def calculate_and_print_route(start_state):
    current_state = start_state
    total_cost = 0
    route = [current_state]

    while current_state[0:2] not in end_coordinates:
        # Get the action to take from the policy
        action = policy.get(current_state)

        # Calculate the new state based on the action
        new_vx = current_state[2] + action[0]
        new_vy = current_state[3] + action[1]

        # Check if the velocity change fails (20% chance)
        if random.random() > 0.2:
            # Velocity change successful
            new_x = current_state[0] + new_vx
            new_y = current_state[1] + new_vy
        else:
            # Velocity change failed, keep the previous velocity
            new_x = current_state[0] + current_state[2]
            new_y = current_state[1] + current_state[3]

        # Check if the new state hits a wall
        if hits_wall(current_state[0], current_state[1], new_x, new_y):
            reward = -10  # Penalty for hitting a wall
            route.append(current_state)
        else:
            reward = calculate_reward(current_state, (new_x, new_y, new_vx, new_vy))
            route.append((new_x, new_y, new_vx, new_vy))  # Add the new state to the route
            # Update the current state
            current_state = (new_x, new_y, new_vx, new_vy)

        total_cost += reward

    return total_cost, route


# Function to visualize the route taken by the car on the grid
def visualize_route(route):
    grid = [['-' for _ in range(15)] for _ in range(10)]

    # Update grid with walls
    for x, y in wall_coordinates:
        grid[x][y] = 'W'

    #Update grid with starting point
    for x,y in start_coordinates:
        grid[x][y]= '\033[1;93mS\033[0m'

    # Update grid with ending points
    for x, y in end_coordinates:
        grid[x][y] = '\033[1;92mE\033[0m'

    # Initialize previous state to compare for failures
    previous_state = None

    # Update grid with the route
    for state in route:
        x, y, _, _ = state
        grid[x][y] = '\033[1;94mC\033[0m'  # Symbol for the car's current position
        if state == previous_state:
          grid[x][y] = '\033[1;30;101mC\033[0m'  # Mark failed move
        previous_state = state

    # Print the grid
    for row in grid:
        print(' '.join(row))


# Example usage:

# Initial State
# The initial state includes:
# -- the (x, y) starting coordinates ( They can be chosen randomly out of the starting cells or manually updated)
# -- the (vx, vy) coordinates for the velocity that are set to (0, 0)

current_state = random.choice(start_coordinates) + (0, 0)
#current_state = (4, 1, 0, 0)
cost, route = calculate_and_print_route(current_state)
print("Cost of the path:", cost)
print("\nPath:", route)
print("\nPath Visualization:\n")
visualize_route(route)  # Visualize the route on the grid