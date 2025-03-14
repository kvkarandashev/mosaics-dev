# For automatically transferring changes between mosaics and mosaics-dev.
import os
import shutil
import subprocess
import sys
from copy import deepcopy

synchronized_root_files = [
    "README.md",
    "Makefile",
    "pyproject.toml",
    "LICENSE",
    ".gitignore",
    ".pre-commit-config.yaml",
]

synchronized_folders = ["examples", "tests", "mosaics"]

ignored_template_files = ["tests/submit_all_tests.sh"]


def missing_in_other_list(compared_list, other_list):
    output = []
    for i in compared_list:
        if i not in other_list:
            output.append(i)
    return output


def listed_files(root_dir=None, ignored_files=None):
    if root_dir is not None:
        old_dir = os.getcwd()
        os.chdir(root_dir)
    synchronized_files = deepcopy(synchronized_root_files)
    for synchronized_folder in synchronized_folders:
        synchronized_files += (
            subprocess.check_output(["git", "ls-files", synchronized_folder]).decode().split()
        )
    existing_synchronized_files = []
    for file in synchronized_files:
        if os.path.isfile(file):
            existing_synchronized_files.append(file)
    if root_dir is not None:
        os.chdir(old_dir)
    if ignored_files:
        for ignored_file in ignored_files:
            if ignored_file in existing_synchronized_files:
                i = existing_synchronized_files.index(ignored_file)
                del existing_synchronized_files[i]
    return existing_synchronized_files


def main(template_repository, updated_repository):
    super_root_dir = "../../"

    # List all files to be synchronized.
    template_files = listed_files(
        super_root_dir + template_repository, ignored_files=ignored_template_files
    )

    updated_files = listed_files(super_root_dir + updated_repository)

    updated_repo_missing = missing_in_other_list(template_files, updated_files)
    templated_repo_missing = missing_in_other_list(updated_files, template_files)

    if updated_repo_missing or templated_repo_missing:
        if templated_repo_missing:
            print("Present in " + updated_repository + ", not in " + template_repository + ":")
            print("\n".join(templated_repo_missing))
            print()
        if updated_repo_missing:
            print("Present in " + template_repository + ", not in " + updated_repository + ":")
            print("\n".join(updated_repo_missing))
            print()
        print(
            """Resolve by:
            - renaming files in mosaics to match their names in mosaics-dev via 'git mv';
            - creating new files in mosaics via 'touch' and 'git add' commands;
            - deleting obsolete files in mosaics-dev via 'git remove';
            - deleting files in mosaics-dev that are not to be commited yet (can be restored with 'git restore');
        """
        )
        quit()

    answer = input(
        "Filenames in both repositories match. Do you want to proceed with updating files in "
        + updated_repository
        + " based on "
        + template_repository
        + "? (y/N)\n"
    )
    if answer != "y":
        print("Aborted")

    os.chdir(super_root_dir)
    for file in template_files:
        shutil.copyfile(template_repository + "/" + file, updated_repository + "/" + file)


if __name__ == "__main__":
    template_repository = "mosaics-dev"
    updated_repository = "mosaics"

    args = sys.argv

    if (len(args) > 1) and (args[1] == "--reverse"):
        template_repository, updated_repository = updated_repository, template_repository
    main(template_repository, updated_repository)
