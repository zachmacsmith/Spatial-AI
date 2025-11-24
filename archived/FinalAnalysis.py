import pandas as pd


def read_csv_with_duration_aggregation(file_path):
   
    # Read the first row to get the constant
    with open(file_path, 'r') as f:
        constant = float(f.readline().strip())
   
    # Read the rest of the data starting from the second row
    df = pd.read_csv(file_path,
                     skiprows=1,
                     names=['frames', 'action', 'object'],
                     dtype={'frames': 'float64', 'action': 'str', 'object': 'str'})
   
    # Calculate duration for each row (next_frame - current_frame)
    df['duration'] = df['frames'].shift(-1) - df['frames']
   
    # Remove the last row (it has NaN duration since there's no next frame)
    df = df[:-1]
   
    # Aggregate durations by action
    duration_by_action = df.groupby('action')['duration'].sum().to_dict()
   
    # Aggregate durations by object
    duration_by_object = df.groupby('object')['duration'].sum().to_dict()
   
    # Multiply totals by constant
    final_action_totals = {action: total * constant for action, total in duration_by_action.items()}
    final_object_totals = {obj: total * constant for obj, total in duration_by_object.items()}
   
    return constant, final_action_totals, final_object_totals, df


# Example usage
file_path = 'your_data.csv'
constant, action_totals, object_totals, processed_data = read_csv_with_duration_aggregation(file_path)


print(f"\nTotal time for each action:")
total_action_time = sum(action_totals.values())
for action, total_frames in action_totals.items():
    print(f"  {action}: {(total_frames / total_action_time) * 100}%")


print(f"\nTotal time used per object:")
for obj, total_frames in object_totals.items():
    print(f"  {obj}: {total_frames}")