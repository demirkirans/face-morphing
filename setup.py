import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="face-morphing", # Replace with your own username
    version="0.0.1",
    author="Serhat Demirkiran",
    author_email="serhat.demirkiran@gmail.com",
    description="A tool for morphing human faces",
    keywords="face morphing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/demirkirans/face-morphing",
    packages=setuptools.find_packages(),
    license='MIT',
    install_requires=[
        'opencv',
        'numpy',
        'docopt',
        'moviepy',
        'dlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)