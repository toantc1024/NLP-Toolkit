import subprocess

# Run pip freeze and capture the output
output = subprocess.check_output(['pip', 'freeze'], shell=True).decode('utf-8')

# Filter out lines containing "file://"
filtered_output = '\n'.join(line for line in output.splitlines() if 'file://' not in line)

# Print the cleaned output
print(filtered_output)