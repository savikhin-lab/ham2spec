FROM rust:latest

# Install build-time dependencies, remove all the other cruft afterwards
RUN apt-get update && apt-get install -y valgrind libopenblas-dev gfortran python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --user numpy

# Cache the Rust dependencies so they don't download on every recompile
WORKDIR /ham2spec
COPY Cargo.toml .
RUN mkdir src && touch src/lib.rs && cargo vendor

# Copy the code over
COPY src/ ./src/ 
COPY examples/ ./examples/

# Compile the example
RUN RUSTFLAGS='-C force-frame-pointers=y' cargo build --example multiple_broadened_spectra --release