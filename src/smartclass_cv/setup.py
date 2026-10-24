from setuptools import find_packages, setup

package_name = 'smartclass_cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', [
            'models/best.onnx',
            'models/best.pt',
            'models/face.pt',
            'models/activity.onnx',
            'models/expression.onnx',
            'models/face_landmarker.task',
            'models/hand_landmarker.task',
            'models/gesture_lstm.h5',
            'models/gesture_lstm.onnx',
        ]),
    ],
    package_data={'': ['py.typed']},
    install_requires=[
        'setuptools',
        'torch',
        'ultralytics',
    ],
    zip_safe=True,
    maintainer='agung',
    maintainer_email='agung@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'frame = smartclass_cv.frame_node:main',
            'detection = smartclass_cv.detection_node:main',
            'detection_debug = smartclass_cv.detection_node_instrumented:main',
            'face = smartclass_cv.face_node:main',
            'recognition = smartclass_cv.recognition_node:main',
            'recognition_debug = smartclass_cv.recognition_baseline_probe:main',
            'expression = smartclass_cv.expression_node:main',
            'expression_debug = smartclass_cv.expression_baseline_probe:main',
            'merge = smartclass_cv.merge_node:main',
            'merge_test = smartclass_cv.merge_node_test:main',
            'attendance = smartclass_cv.attendance_node:main',
            'activity = smartclass_cv.activity_node:main',
            'gesture = smartclass_cv.gesture_node:main',
            'activate = smartclass_cv.wakeon_node:main'
        ,
            'gesture_debug = smartclass_cv.gesture_baseline_probe:main',
        ],
    },
)
