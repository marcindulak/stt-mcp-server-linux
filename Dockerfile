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
# The usermod below creates a security risk due to non least-privilege grants.
# The input group created by udev can have different id on the host and inside of the container
# and we don't know the id on the host, so grant USER membership in a group range.
RUN set -eux && \
    for gid in $(seq 990 999); do \
        if ! getent group "$gid" >/dev/null; then \
            groupadd -g "$gid" "group$gid"; \
        fi \
    done
RUN usermod -aG $(seq -s, 990 999) ${USER}
RUN cat /etc/passwd
RUN cat /etc/group

ENV XDG_CACHE_HOME=/.whisper
RUN mkdir ${XDG_CACHE_HOME} && chown -R ${USER}:${USER} ${XDG_CACHE_HOME} && chmod 0700 ${XDG_CACHE_HOME}
ENV TMUX_TMPDIR=/.tmux
RUN mkdir ${TMUX_TMPDIR} && chown -R ${USER}:${USER} ${TMUX_TMPDIR} && chmod 0700 ${TMUX_TMPDIR}

WORKDIR /app
RUN chown -R ${USER}:${USER} /app && chmod 0700 /app
USER ${USER}

RUN whoami
RUN id

COPY requirements.txt .
RUN cd /home/${USER} && python3 -m venv --system-site-packages venv && . venv/bin/activate
# Use two step installation as a workaround to install cpu-only torch
# See https://github.com/huggingface/transformers/issues/39780
RUN . /home/${USER}/venv/bin/activate && python -m pip install torch==2.* --index-url https://download.pytorch.org/whl/cpu
RUN . /home/${USER}/venv/bin/activate && python -m pip install -r requirements.txt
RUN echo "if [ -f /home/${USER}/venv/bin/activate  ]; then . /home/${USER}/venv/bin/activate; fi" >> /home/${USER}/.bashrc
COPY stt_mcp_server_linux.py .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep -f "stt_mcp_server_linux.py" || exit 1

CMD ["/home/nonroot/venv/bin/python", "/app/stt_mcp_server_linux.py"]
