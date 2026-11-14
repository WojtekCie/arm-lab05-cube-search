import os
from glob import glob

from setuptools import setup

package_name = 'cube_search'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wojciech',
    maintainer_email='wojtekemcie@gmail.com',
    description='Autonomiczne wyszukiwanie kostek ArUco (Lab 05 ARM): detekcja, mapa pokrycia '
                'kamery, explore_lite, akcja /search_cubes.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector = cube_search.aruco_detector:main',
            'search_server = cube_search.search_server:main',
            'search_client = cube_search.search_client:main',
            'pose_error_logger = cube_search.pose_error_logger:main',
        ],
    },
)
