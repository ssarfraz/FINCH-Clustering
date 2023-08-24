import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setuptools.setup(
    name="finch-clust",
    version="0.2.0",
    author="Saquib Sarfraz",
    author_email="saquibsarfraz@gmail.com",
    description="FINCH - First Integer Neighbor Clustering Hierarchy: A parameter-free fast clustering algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['finch'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: Free for non-commercial use",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="finch, finch clustering, clustering, hierarchical clustering",
    install_requires=[
        "scipy",
        "scikit-learn",
        "numpy"
    ],
    extras_require={'ann': ["numpy==1.21", "pynndescent"]},
    project_urls={
        "Repository": "https://github.com/ssarfraz/FINCH-Clustering",
        "Publication": "https://openaccess.thecvf.com/content_CVPR_2019/html/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.html"
    }
)
