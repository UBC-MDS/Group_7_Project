# Author : Gretel Tan, Riya Eliza, Charles Xu, Yan Zeng
FROM quay.io/jupyter/minimal-notebook:2023-11-19 

RUN conda install -y matplotlib>=3.8.0 \
pandas>=2.1.1 

RUN pip install ucimlrepo==0.0.3