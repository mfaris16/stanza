import subprocess


def output_subtrees(input_file, output_file):
    # TODO: maybe can convert this to use the python tree?
    # TODO: -output doesn't exist...
    cmd = ["java", "edu.stanford.nlp.trees.OutputSubtrees", "-input", input_file, "-output", output_file]
    subprocess.run(cmd)

def convert_asdf:
    TODO
