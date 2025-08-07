#!/bin/bash

# Define variables for the pods and ports
SERVICE_NAME="openchat-vllm"
NAMESPACE="wenglab-vs"
LOCAL_PORT=8000
REMOTE_PORT=8000

# Function to check if kubectl is installed
check_kubectl() {
    if ! command -v kubectl &>/dev/null; then
        echo "kubectl not found, installing locally..."
        download_kubectl
    else
        echo "kubectl is already installed."
    fi
}

# Function to check if curl is installed
check_curl() {
    if ! command -v curl &>/dev/null; then
        echo "curl not found, installing locally..."
        download_curl
    else
        echo "curl is already installed."
    fi
}

# Function to download and install kubectl locally
download_kubectl() {
    kubectl_url="https://storage.googleapis.com/kubernetes-release/release/v1.24.0/bin/linux/amd64/kubectl"
    # Use wget if curl isn't available
    if command -v wget &>/dev/null; then
        wget "$kubectl_url" -O kubectl
    else
        curl -LO "$kubectl_url"
    fi
    chmod +x kubectl
    mv kubectl ~/bin/
    echo "kubectl installed in ~/bin"
}

# Function to download and install curl locally
download_curl() {
    # Check if wget is available to download curl
    if command -v wget &>/dev/null; then
        echo "Downloading curl precompiled binary using wget..."
        wget https://curl.se/download/curl-7.81.0-linux-x86_64.tar.gz
    else
        echo "No curl or wget available. Cannot proceed with the installation."
        exit 1
    fi

    # Check if download was successful
    if [ $? -ne 0 ]; then
        echo "Failed to download curl binary
