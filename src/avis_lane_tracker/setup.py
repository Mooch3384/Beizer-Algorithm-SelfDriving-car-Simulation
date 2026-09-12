import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'avis_lane_tracker'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Amir',
    maintainer_email='amir@todo.todo',
    description='ROS 2 Python package for real-time online lane tracking, IPM, trajectory plotting, and local mapping in AvisEngine simulation.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_tracker_node = avis_lane_tracker.lane_tracker_node:main',
            'avis_sim_bridge = avis_lane_tracker.avis_sim_bridge:main',
        ],
    },
)
