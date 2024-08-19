import csv
import glob

def extract_atomic_commands():
    csv_file = open('omp_atomic_commands.csv', 'w', newline='')
    writer = csv.writer(csv_file, delimiter='@')

    file_count = 0
    line_count = 0

    for filename in glob.glob('*/*.*'):
        file_count += 1

        with open(filename, 'r') as file:
            current_line = 0
            command = ''

            for line in file:
                current_line += 1

                if line.startswith('#pragma omp atomic'):
                    line_count += 1

                    next_line = file.readline()
                    current_line += 1

                    if not next_line:  # Check if next line exists
                        continue  # Skip if no next line

                    # Append pattern line to command
                    command += line.strip() + '\n'

                    if '{' in next_line:  # Handle atomic block with {}
                        command += next_line.strip()

                        while True:
                            next_line = file.readline()
                            if not next_line:
                                break

                            current_line += 1

                            if '}' in next_line:
                                command += next_line.strip() + '\n'
                                break

                            command += next_line.strip() + '\n'

                    else:  # Handle single-line atomic command
                        command += next_line.strip() + '\n'

                    writer.writerow([filename, current_line, command])

    csv_file.close()

    print(f"Total files scanned: {file_count}")
    print(f"Total atomic commands found: {line_count}")

if __name__ == '__main__':
    extract_atomic_commands()

