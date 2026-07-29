from setuptools import find_packages,setup
from typing import List

hypen = "-e ."
def get_requirements(file_path:str)-> List[str]:
    requiremnts = [ ]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [r.replace("\n", "") for r in requirements]
        if hypen in requirements:
            requirements.remove(hypen)
    return requirements


setup(
name = "ML_Project",
version = "0.0.1",
author = "Jash",
author_email = "jasujavangula@gmail.com",
packages = find_packages(),
install_requires = get_requirements("requirements.txt")
)
