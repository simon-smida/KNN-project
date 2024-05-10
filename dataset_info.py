import sys
import random

def select_lines_with_ids(filename):
    id_lines = {}

     
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_parts = parts[1].split('/')
                if len(id_parts) > 0:
                    id = id_parts[0]
                    if id in id_lines:
                        id_lines[id].append(line.strip())
                    else:
                        id_lines[id] = [line.strip()] 
                        

    filtered_ids = [id for id, lines in id_lines.items() if len(lines) > 100]

    selected_lines = []
    for id in filtered_ids:
        lines = id_lines[id]
        if len(lines) > 100:
            selected_lines.extend(random.sample(lines, 100))
        else:
            selected_lines.extend(lines) 

    random.shuffle(selected_lines)

    with open('selected_lines.txt', 'w') as outfile:
        for line in selected_lines:
            outfile.write(f"{line}\n")

    print(f"Successfully created 'selected_lines.txt' with exactly 100 samples per selected ID.")

    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python select_lines.py <filename>")
        sys.exit(1)
        
    filename = sys.argv[1]

    select_lines_with_ids(filename)