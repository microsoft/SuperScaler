# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command import build_ext, install_lib

with open("README.md", "r") as fh:
    long_description = fh.read()

global CURRENT_DIR
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# set the (lib name, src dir from build, dst dir to install) during install_lib
global LIB_INSTALL_LIST
LIB_INSTALL_LIST = [
    ('libsuperscaler_pywrap.so', 'lib/', 'superscaler/lib/')
]


class CMakeExtension(Extension):
    '''Wrapper for Extension'''
    def __init__(self, name):
        '''use cmake and cmake --build to build extension'''
        super().__init__(name, sources=[])


class BuildExt(build_ext.build_ext):
    '''Wrapper for build_ext to support cmake to build extensions'''

    def build_extension(self, ext):
        '''Override the method to support cmake'''
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
        else:
            super().build_extension(ext)

    def build_cmake_extension(self, ext):
        '''define the cmake and --build process'''
        global BUILD_DIR
        BUILD_DIR = os.path.join(CURRENT_DIR, self.build_temp)

        # make dir for build
        if not os.path.exists(BUILD_DIR):
            os.mkdir(BUILD_DIR)

        # set command and args of cmake and make
        nproc = os.cpu_count()
        cmake_cmd = ['cmake', CURRENT_DIR]
        cmake_args = []
        build_cmd = ['cmake', '--build', '.']
        build_args = ['-j', str(nproc)]

        # execute camke and build to build the extension
        subprocess.check_call(cmake_cmd + cmake_args, cwd=BUILD_DIR)
        if not self.dry_run:
            subprocess.check_call(build_cmd + build_args, cwd=BUILD_DIR)


class InstallLib(install_lib.install_lib):
    '''Wrapper for install_lib to install libs into python package'''
    def install(self):
        outfiles = super().install()

        # copy every items in LIB_INSTALL_LIST from src to dst directory
        for lib, src, dst in LIB_INSTALL_LIST:
            dst_dir = os.path.join(CURRENT_DIR, self.install_dir, dst)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            _dst, _ = self.copy_file(os.path.join(BUILD_DIR, src, lib),
                                     os.path.join(dst_dir, lib))
        outfiles.append(_dst)


if __name__ == '__main__':

    packages = find_packages('src')

    tensorflow_requires = ['tensorflow>=1.15,<2']
    tensorflow_cpu_requires = ['tensorflow-cpu>=1.15,<2']
    tensorflow_gpu_requires = ['tensorflow-gpu>=1.15,<2']

    all_frameworks_requires = tensorflow_requires

    setup(
        name='superscaler',
        version='0.1',
        description='A distributed training platform for deep learning',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author='SuperScaler Team',
        author_email='superscaler@microsoft.com',
        url="https://github.com/microsoft/superscaler",
        zip_safe=False,
        python_requires='>=3.6,<3.8',
        install_requires=[
            'protobuf',
            'bitmath',
            'humanreadable',
            'PyYAML',
            'scipy',
            'networkx',
            'matplotlib',
            'numpy',
        ],
        extras_require={
            'tensorflow': tensorflow_requires,
            'tensorflow-cpu': tensorflow_cpu_requires,
            'tensorflow-gpu': tensorflow_gpu_requires,
            'all-frameworks': all_frameworks_requires
        },
        ext_modules=[CMakeExtension('libsuperscaler_pywrap')],
        cmdclass={
            'build_ext': BuildExt,
            'install_lib': InstallLib,
        },
        packages=packages,
        package_dir={'superscaler': 'src/superscaler'},
        license='MIT License',
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ]
    )
