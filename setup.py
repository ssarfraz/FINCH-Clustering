import setuptools
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="finch-clust",
    version="0.2.2",  # bump each release
    python_requires=">=3.8",
    packages=["finch"],  # or: packages=setuptools.find_packages()
    include_package_data=True,       # if you ship non-.py files via MANIFEST.in
    # license="...",                 # set this if you have a specific license
    # url="https://github.com/ssarfraz/FINCH-Clustering",
    install_requires=["scipy", "scikit-learn", "numpy"],
    extras_require={"ann": ["pynndescent"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    project_urls={
        "Repository": "https://github.com/ssarfraz/FINCH-Clustering",
        "Publication": "https://openaccess.thecvf.com/content_CVPR_2019/html/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.html"
    },
    description="FINCH - First Integer Neighbor Clustering Hierarchy: A parameter-free fast clustering algorithm.",
    author="Saquib Sarfraz",
    author_email="saquibsarfraz@gmail.com",
)
