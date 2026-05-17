FROM quay.io/jupyterhub/jupyterhub:4.0

# dockerspawner depends on the docker SDK transitively, but pin it
# explicitly so the version used by jupyterhub_config.py self-inspection
# is stable regardless of upstream transitive resolution changes.
RUN python3 -m pip install --no-cache-dir \
    dockerspawner==14.0.0 \
    jupyterhub-nativeauthenticator==1.3.0 \
    docker==7.1.0

CMD ["jupyterhub", "-f", "/srv/jupyterhub/jupyterhub_config.py"]