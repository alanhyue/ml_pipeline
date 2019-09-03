import subprocess
import sys

def section(title):
    print("="*50)
    print('{:^50s}'.format(title))
    print("="*50)

section('TESTING')
subprocess.run(["python", "-m", "unittest", "discover", "tests/", "-v"])

section('UPLOAD')
import ml_pipeline
print("Current version:", ml_pipeline.__version__)
ver = input("Enter new version number: ")
with open('ml_pipeline/__version__.py', 'w') as fout:
    fout.write('__version__="{}"'.format(ver))

print()
print("Compiling..")
subprocess.run(['python', 'setup.py', 'sdist'])

print()
print('Uploading..')
subprocess.run(['twine', 'upload', 'dist/ml_pipeline-{}.tar.gz'.format(ver)])

