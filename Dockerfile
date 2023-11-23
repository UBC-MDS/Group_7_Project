# Author : Gretel Tan, Riya Eliza, Charles Xu
FROM quay.io/jupyter/minimal-notebook:2023-11-19 

# RUN conda install -y -c conda-forge pandas pip scikit-learn altair altair_saver\
#     jupyter_contrib_nbextensions jupyter-book matplotlib pyppeteer 
# RUN pip install docopt-ng vl-convert-python ucimlrepo==0.0.3

RUN conda install -y matplotlib \
pandas\
scikit-learn\
altair \
bzip2\
ca-certificates=2023.7.22 \
libexpat \
libffi \
libsqlite \
libzlib\ 
openssl\
pip \
python\
setuptools \ 
tk \
tzdata=2023c \
wheel\
xz \
vl-convert-python  \
vegafusion \
vegafusion-python-embed  \
vegafusion-jupyter  

RUN pip install ucimlrepo==0.0.3