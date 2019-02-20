from setuptools import setup, find_packages
# import os, sys
# import subprocess

#
# # import version
# with open('libwallerlab/_version.py','r') as f:
#     exec(f.read())
# assert '__version__' in locals()
# __version__ = 0.1


# Get the current git commit hash string and store it as libwallerlab.__gitcommithash__
# try:
#     git_commit_string = subprocess.check_output("git rev-parse HEAD", shell=True)
#     git_commit_string = git_commit_string[:-1].decode('ascii')
# except:
#     git_commit_string = ''
# if sys.platform == 'win32':
#     os.system("""echo __gitcommithash__='%s' > libwallerlab/_gitcommithash.py""" % (git_commit_string,))
# else:
#     os.system("""echo "__gitcommithash__='%s'" > libwallerlab/_gitcommithash.py""" % (git_commit_string,))



setup( name             = 'llops'
     , version          = '0.1'
     , description      = 'Low-level Operators with a backend selector'
     , license          = 'BSD'
     , packages         = find_packages()
     , py_modules       = ['llops']
     , install_requires = ['numpy', 'matplotlib', 'scipy', 'scikit-image', 'imageio', 'pillow']
     )
