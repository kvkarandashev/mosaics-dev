# For conveniently comparing where two mosaics forks differ.
import subprocess
import sys

from public_repo_update import listed_files


def identical_files(file1, file2):
    try:
        return len(subprocess.check_output(["diff", file1, file2])) == 0
    except subprocess.CalledProcessError:
        return False


def compare_repos(repo_dir1, repo_dir2):
    files1 = listed_files(repo_dir1)
    files2 = listed_files(repo_dir2)
    common_files = []
    repo1_only = []
    for file1 in files1:
        if file1 in files2:
            common_files.append(file1)
        else:
            repo1_only.append(file1)
    repo2_only = []
    for file2 in files2:
        if file2 not in files1:
            repo2_only.append(file2)
    for repo_only, repo_dir in zip([repo1_only, repo2_only], [repo_dir1, repo_dir2]):
        print("present only in", repo_dir, ":\n", "\n".join(repo_only))
    differing_files = []
    for file in common_files:
        if not identical_files(repo_dir1 + "/" + file, repo_dir2 + "/" + file):
            differing_files.append(file)
    if differing_files:
        print("differing_files:\n", "\n".join(differing_files))


if __name__ == "__main__":
    assert len(sys.argv) == 3
    compare_repos(*sys.argv[1:])
