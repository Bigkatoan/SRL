from setuptools import find_packages, setup


package_name = "srl_inference_pkg"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/inference.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SRL User",
    maintainer_email="maintainer@example.com",
    description="Starter ROS 2 ament package for SRL inference deployment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "srl_inference_node = srl_inference_pkg.inference_node:main",
        ],
    },
)