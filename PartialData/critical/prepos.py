import sys
import re

def process_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if re.search(r'\b((\+\+|--)\w+|\w+(\+\+|--)\b)', line):
            print(f"Line containing pre- or post-increment/decrement: {line}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python increment_decrement_detector.py <C_program_file>")
        exit()

    filename = sys.argv[1]
    process_file(filename)

