from complexity_metrics import get_gmacs_and_params, get_runtime
import codes.models.modules.adnet as adnet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--write_path", type=str, default="./", help="Path to write the readme.txt file")
args = parser.parse_args()
write_path=args.write_path

# Load a pytorch model
model = adnet.ADNetv2(6, 5, 64, 32)
model.eval()

# Calculate MACs and Parameters

height = 1060
width = 1900

scale = 4  # You can try to modify this to 1 if your GPU allowed.

total_macs, total_params = get_gmacs_and_params(model, input_size=(1, 6, height // scale, width // scale))
mean_runtime = get_runtime(model, input_size=(1, 6, height // scale, width // scale))

total_macs = total_macs * scale * scale
mean_runtime = mean_runtime * scale * scale

print(total_macs)
print(total_params)
print(mean_runtime)

# Print model statistics to txt file
with open(write_path + 'readme.txt', 'w') as f:
    f.write("runtime per image [s] : " + str(mean_runtime))
    f.write('\n')
    f.write("number of operations [GMAcc] : " + str(total_macs))
    f.write('\n')
    f.write("number of parameters  : " + str(total_params))
    f.write('\n')
    f.write("Other description: Toy Model for demonstrating example code usage.")
# Expected output of the readme.txt for ToyHDRModel should be:
# runtime per image [s] : 0.013018618555068967
# number of operations [GMAcc] : 20.146042
# number of parameters  : 8243
# Other description: Toy Model for demonstrating example code usage.

print("You reached the end of the calculate_ops_example.py demo script. Good luck participating in the NTIRE 2022 HDR Challenge!")

