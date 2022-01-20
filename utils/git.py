def get_branch_name():
    with open("../.git/HEAD", "r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def get_commit_hash(branch_name):
    with open("../.git/refs/heads/" + branch_name) as f:
        commit_hash = f.read()[0:6]

    return commit_hash
