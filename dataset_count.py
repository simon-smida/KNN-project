import sys

def count_ids_from_file(filename):
    id_count = {}

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_parts = parts[1].split('/')
                if len(id_parts) > 0:
                    id = id_parts[0]
                    if id in id_count:
                        id_count[id] += 1
                    else:
                        id_count[id] = 1

    sorted_counts = sorted(id_count.items(), key=lambda x: x[1], reverse=True)

    for id, count in sorted_counts:
        print(f"ID: {id}, Count: {count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_ids.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    count_ids_from_file(filename)