# Seems like the best image to use as no Pytorch image available
FROM jupyter/scipy-notebook

# Cleaning up base file directory
RUN rmdir work

# Installing missing packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copying code
COPY --chown=jovyan:users src /home/jovyan/src

# Environment niceties
ENV JUPYTER_ENABLE_LAB=yes
