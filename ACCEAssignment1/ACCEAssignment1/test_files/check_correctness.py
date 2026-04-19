from sys import argv

def get_values(text):
    for line in text.readlines():
        if line.startswith("Result:"):
            return list(map(float, line.split(':')[1].split(',')))

def read_files(file1, file2):
    diff = {}
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            d1 = get_values(f1)
            d2 = get_values(f2)
            diff["iterations"] = d1[0] - d2[0]
            diff["max_flow_i"] = d1[1] - d2[1]
            diff["max_flow_a"] = (d1[2] - d2[2]) / (d1[2] + d2[2])
            diff["max_level"] = (d1[3] - d2[3]) / (d1[3] + d2[3])
            diff["total_rain"] = (d1[4] - d2[4]) / (d1[4] + d2[4])
            total_water = d1[4] + d2[4] + d1[5] + d2[5]
            diff["final_water"] = (d1[5] - d2[5]) / total_water
            diff["final_outflow"] = (d1[6] - d2[6]) / total_water
        return diff
    except FileNotFoundError:
        print("One of your file paths is incorrect.")
        exit(1)


def is_valid(differences):
    #    and differences["max_flow_i"] == 0 \
    return differences["iterations"] == 0 \
       and abs(differences["max_flow_a"]) < 0.01 \
       and abs(differences["max_level"]) < 0.01 \
       and abs(differences["total_rain"]) < 0.0001 \
       and abs(differences["final_water"]) < 0.0001 \
       and abs(differences["final_outflow"]) < 0.0001


if __name__ == "__main__":
    if len(argv) != 3:
        print("You must supply a reference file and your own output to compare.")
        exit(1)

    differences = read_files(argv[1], argv[2])
    if is_valid(differences):
        print("Your output matches the reference.")
    else:
        print("Your output does NOT(!) match the reference.")