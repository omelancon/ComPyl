from setuptools import setup, find_packages

setup(name='compyl',
      packages=find_packages(exclude=("tests",)),
      version='0.1.3',
      description='Python lexing-parsing tool',
      author='Olivier Melancon',
      author_email='ol.melancon@gmail.com',
      url='https://github.com/omelancon/ComPyl',
      download_url='https://github.com/omelancon/ComPyl/archive/v0.1.3.tar.gz',
      install_requires=[
        'dill',
      ],
      keywords='lexer lexing parser parsing compiler',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Software Development :: Compilers',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
      ]
      )
