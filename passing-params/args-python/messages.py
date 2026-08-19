import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--message1", type=str)
parser.add_argument("--message2", type=str)
args = parser.parse_args()
print(args.message1)
print(args.message2)
