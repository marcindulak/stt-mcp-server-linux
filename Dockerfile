FROM debian:trixie-slim

ENV USER=nonroot

# Install base dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libasound2-dev \
    portaudio19-dev \
    procps \
    tmux \
    python3 \
    python3-pip \
    python3-venv \
    udev

# Install vosk dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-certifi \
    python3-cffi \
    python3-evdev \
    python3-idna \
    python3-pycparser \
    python3-tqdm \
    python3-urllib3 \
    python3-websockets

# Install whisper dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-charset-normalizer \
    python3-filelock \
    python3-fsspec \
    python3-jinja2 \
    python3-llvmlite \
    python3-markupsafe \
    python3-more-itertools \
    python3-mpmath \
    python3-networkx \
    python3-numba \
    python3-numpy \
    python3-regex \
    python3-requests \
    python3-setuptools \
    python3-sympy \
    python3-typing-extensions

# Install test dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-iniconfig \
    python3-mypy \
    python3-pluggy \
    python3-pygments \
    python3-pytest

# Set the UID/GID of the user:group to the IDs of the user using this Dockerfile
ARG USER=nonroot
ARG GROUP=nonroot
ARG UID=1000
ARG GID=1000
RUN echo user:group ${USER}:${GROUP}
RUN echo uid:gid ${UID}:${GID}
RUN getent group ${GROUP} || groupadd --non-unique --gid ${GID} ${GROUP}
RUN getent passwd ${USER} || useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USER}
RUN if [ "${GID}" != "1000" ] || [ "${UID}" != "1000" ]; then \
      groupmod --non-unique --gid ${GID} ${GROUP} && \
      usermod --uid ${UID} --gid ${GID} ${USER} && \
      chown -R ${UID}:${GID} /home/${USER}; \
    fi

RUN usermod -aG audio ${USER}
# The /dev/input group owner ID may differ outside/inside the container.
# For the keyboard detection to work inside of the container,
# the user inside of the container must be the member
# of the /dev/input group ID present outside of the container.
ARG INPUT_GID=900
RUN getent group ${INPUT_GID} || groupadd --non-unique --gid ${INPUT_GID} ${INPUT_GID}
RUN usermod -aG ${INPUT_GID} ${USER}
# The membership of the user in the /dev/input group ID present inside of the container
# does not matter for this application, but let's still make the ${USER} to be member of this group
RUN usermod -aG $(getent group input | cut -d: -f3) ${USER}

RUN cat /etc/passwd
RUN cat /etc/group

ENV XDG_CACHE_HOME=/.whisper
RUN mkdir ${XDG_CACHE_HOME} && chown -R ${USER}:${USER} ${XDG_CACHE_HOME} && chmod 0700 ${XDG_CACHE_HOME}
ENV TMUX_TMPDIR=/.tmux
RUN mkdir -p ${TMUX_TMPDIR}/tmux-${UID} && chown -R ${USER}:${USER} ${TMUX_TMPDIR} && chmod 0700 ${TMUX_TMPDIR}

WORKDIR /app
RUN chown -R ${USER}:${USER} /app && chmod 0700 /app
USER ${USER}

RUN whoami
RUN id

COPY requirements.txt .
RUN cd /home/${USER} && python3 -m venv --system-site-packages venv && . venv/bin/activate
# Debian's python3-torch is slow
# Use two step installation as a workaround to install cpu-only torch
# See https://github.com/huggingface/transformers/issues/39780
RUN . /home/${USER}/venv/bin/activate && python -m pip install torch==2.* --index-url https://download.pytorch.org/whl/cpu
RUN . /home/${USER}/venv/bin/activate && python -m pip install -r requirements.txt
RUN echo "if [ -f /home/${USER}/venv/bin/activate  ]; then . /home/${USER}/venv/bin/activate; fi" >> /home/${USER}/.bashrc
COPY stt_mcp_server_linux.py .

# Avoid overloading the system with too many openblas threads
ENV OPENBLAS_NUM_THREADS=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep -f "stt_mcp_server_linux.py" || exit 1

CMD ["/home/nonroot/venv/bin/python", "/app/stt_mcp_server_linux.py"]
