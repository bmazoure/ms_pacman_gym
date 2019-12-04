from setuptools import setup, find_packages

setup(name='pocman_gym',
      version='1.0.0',
      install_requires=['gym'],  # And any other dependencies foo needs,
      include_package_data=True,
      packages=find_packages()
)  