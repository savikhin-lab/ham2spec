alias c := check
alias t := test
alias bd := build-debug
alias br := build-release

# Display available commands
default:
    @just --list --unsorted

# Check
check:
    cargo check

# Run tests
test:
    cargo test --lib

# Build - debug
build-debug:
    cargo build

# Build - release
build-release:
    cargo build --release
