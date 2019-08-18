import os
# for file handling
def absolute_file_path(rel_path):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, rel_path)
