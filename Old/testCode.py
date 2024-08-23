import numpy as np

def distribute_events(total_events, branching_ratios):
    # Calculate the initial number of events for each channel
    events = np.array([total_events * ratio for ratio in branching_ratios])
    
    # Round to the nearest integer
    rounded_events = np.round(events).astype(int)
    
    # Calculate the difference between the total rounded events and the desired total
    difference = total_events - np.sum(rounded_events)
    
    # Adjust the number of events to make sure the total sums up to total_events
    while difference != 0:
        for i in range(len(branching_ratios)):
            if difference == 0:
                break
            if difference > 0 and (rounded_events[i] < events[i] or np.sum(rounded_events) < total_events):
                rounded_events[i] += 1
                difference -= 1
            elif difference < 0 and rounded_events[i] > 0:
                rounded_events[i] -= 1
                difference += 1

    return rounded_events

# Define the branching ratios for each channel
branching_ratios = [0.276, 0.498, 0.226]
total_events = 1005

# Distribute the events based on the branching ratios
event_distribution = distribute_events(total_events, branching_ratios)

# Output the results
for i, ratio in enumerate(branching_ratios):
    print(f"Channel {i+1} ({ratio}): {event_distribution[i]} events")

print(f"Total events: {np.sum(event_distribution)}")
