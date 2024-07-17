import re
from collections import defaultdict


def parse_log_file(file_path):
    # Regular expression to match the specific log format, with optional None for weight.shape and bias.shape
    pattern = re.compile(
        r"torch\.nn\.Conv2d\.forward: input\.shape=torch\.Size\(([^)]+)\) "
        r"weight\.shape=(torch\.Size\(([^)]+)\)|None) bias\.shape=(torch\.Size\(([^)]+)\)|None)\s+"
        r"stride=\([^)]+\) padding=\([^)]+\) dilation=\([^)]+\) groups=\d+\s+"
        r"kernel_size=\(([^)]+)\) in_channels=(\d+) out_channels=(\d+)"
    )

    # Dictionary to store the count of each tuple
    counter = defaultdict(int)

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                input_shape = match.group(1)
                weight_shape = match.group(3) if match.group(2) != "None" else "None"
                bias_shape = match.group(5) if match.group(4) != "None" else "None"
                kernel_size = match.group(6)
                in_channels = match.group(7)
                out_channels = match.group(8)

                # Create a tuple with the extracted values
                key = (
                    input_shape,
                    weight_shape,
                    bias_shape,
                    in_channels,
                    out_channels,
                    kernel_size,
                )
                # Increment the count of this tuple in the dictionary
                counter[key] += 1

    return counter


def print_markdown_table(counter):
    # Define the header
    header = [
        "Input Shape",
        "Weight Shape",
        "Bias Shape",
        "In Channels",
        "Out Channels",
        "Kernel Size",
        "Count",
    ]

    # Create the Markdown table header
    table = "| " + " | ".join(header) + " |\n"
    table += "| " + " | ".join(["---"] * len(header)) + " |\n"

    # Add the rows to the table
    for key, count in counter.items():
        row = list(key) + [str(count)]
        table += "| " + " | ".join(row) + " |\n"

    return table


if __name__ == "__main__":
    # Example usage
    log_file_path = "./run.log"
    result = parse_log_file(log_file_path)
    markdown_table = print_markdown_table(result)
    print(markdown_table)
