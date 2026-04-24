#!/bin/bash
# Echo Agent installer for Linux, macOS, and WSL2.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

REPO_URL_SSH="git@github.com:fuyuxiang/echo-agent.git"
REPO_URL_HTTPS="https://github.com/fuyuxiang/echo-agent.git"
ECHO_HOME="${ECHO_HOME:-$HOME/.echo-agent}"
INSTALL_DIR="${ECHO_INSTALL_DIR:-$ECHO_HOME/echo-agent}"
PYTHON_VERSION="3.11"
BRANCH="main"
RUN_SETUP=true

if [ -t 0 ]; then
    IS_INTERACTIVE=true
else
    IS_INTERACTIVE=false
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-setup)
            RUN_SETUP=false
            shift
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --echo-home)
            ECHO_HOME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Echo Agent Installer"
            echo ""
            echo "Usage: install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-setup      Skip interactive setup wizard"
            echo "  --branch NAME     Git branch to install (default: main)"
            echo "  --dir PATH        Installation directory (default: ~/.echo-agent/echo-agent)"
            echo "  --echo-home PATH  Echo home directory (default: ~/.echo-agent)"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_banner() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│                 Echo Agent Installer                   │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│  Self-hosted AI agent runtime for your own workspace.  │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
}

log_info() {
    echo -e "${CYAN}→${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}!${NC} $1"
}

log_error() {
    echo -e "${RED}x${NC} $1"
}

prompt_yes_no() {
    local question="$1"
    local default="${2:-yes}"
    local prompt_suffix
    local answer=""

    case "$default" in
        [yY]|[yY][eE][sS]|[tT][rR][uU][eE]|1) prompt_suffix="[Y/n]" ;;
        *) prompt_suffix="[y/N]" ;;
    esac

    if [ "$IS_INTERACTIVE" = true ]; then
        read -r -p "$question $prompt_suffix " answer || answer=""
    elif [ -r /dev/tty ] && [ -w /dev/tty ]; then
        printf "%s %s " "$question" "$prompt_suffix" > /dev/tty
        IFS= read -r answer < /dev/tty || answer=""
    fi

    answer="${answer#"${answer%%[![:space:]]*}"}"
    answer="${answer%"${answer##*[![:space:]]}"}"
    if [ -z "$answer" ]; then
        case "$default" in
            [yY]|[yY][eE][sS]|[tT][rR][uU][eE]|1) return 0 ;;
            *) return 1 ;;
        esac
    fi

    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

detect_os() {
    case "$(uname -s)" in
        Linux*)
            OS="linux"
            ;;
        Darwin*)
            OS="macos"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            log_error "Native Windows is not supported."
            log_info "Use WSL2 and run this installer there."
            exit 1
            ;;
        *)
            log_error "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac
    log_success "Detected: $OS"
}

check_git() {
    log_info "Checking Git..."
    if command -v git >/dev/null 2>&1; then
        log_success "$(git --version)"
        return 0
    fi
    log_error "Git not found."
    if [ "$OS" = "macos" ]; then
        log_info "Install it with: xcode-select --install"
    else
        log_info "Install it with your package manager, then rerun this script."
    fi
    exit 1
}

install_uv() {
    log_info "Checking uv..."
    if command -v uv >/dev/null 2>&1; then
        UV_CMD="uv"
        log_success "$(uv --version)"
        return 0
    fi
    if [ -x "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
        log_success "$($UV_CMD --version)"
        return 0
    fi

    log_info "Installing uv..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        if [ -x "$HOME/.local/bin/uv" ]; then
            UV_CMD="$HOME/.local/bin/uv"
        elif command -v uv >/dev/null 2>&1; then
            UV_CMD="uv"
        else
            log_error "uv installed but not found on PATH."
            exit 1
        fi
        log_success "$($UV_CMD --version)"
        return 0
    fi

    log_error "Failed to install uv."
    exit 1
}

check_python() {
    log_info "Checking Python $PYTHON_VERSION..."
    if command -v python3 >/dev/null 2>&1; then
        if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
            PYTHON_PATH="$(command -v python3)"
            log_success "$($PYTHON_PATH --version)"
            return 0
        fi
    fi

    if "$UV_CMD" python find "$PYTHON_VERSION" >/dev/null 2>&1; then
        PYTHON_PATH="$("$UV_CMD" python find "$PYTHON_VERSION")"
        log_success "$($PYTHON_PATH --version)"
        return 0
    fi

    log_info "Installing Python $PYTHON_VERSION via uv..."
    "$UV_CMD" python install "$PYTHON_VERSION"
    PYTHON_PATH="$("$UV_CMD" python find "$PYTHON_VERSION")"
    log_success "$($PYTHON_PATH --version)"
}

clone_repo() {
    mkdir -p "$ECHO_HOME"
    log_info "Preparing repository in $INSTALL_DIR..."

    if [ -d "$INSTALL_DIR/.git" ]; then
        cd "$INSTALL_DIR"
        if [ -n "$(git status --porcelain)" ]; then
            log_warn "Local changes detected in $INSTALL_DIR; skipping update."
            log_info "Clean the repo manually if you want the installer to update it."
        else
            git fetch origin
            git checkout "$BRANCH"
            git pull --ff-only origin "$BRANCH"
            log_success "Repository updated"
        fi
        return 0
    fi

    if [ -e "$INSTALL_DIR" ]; then
        log_error "Install directory exists but is not a git repository: $INSTALL_DIR"
        exit 1
    fi

    log_info "Trying SSH clone..."
    if GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=5" \
        git clone --branch "$BRANCH" "$REPO_URL_SSH" "$INSTALL_DIR" 2>/dev/null; then
        log_success "Cloned via SSH"
        return 0
    fi

    log_info "SSH unavailable, trying HTTPS..."
    git clone --branch "$BRANCH" "$REPO_URL_HTTPS" "$INSTALL_DIR"
    log_success "Cloned via HTTPS"
}

setup_venv() {
    cd "$INSTALL_DIR"
    log_info "Creating virtual environment..."
    rm -rf venv
    "$UV_CMD" venv venv --python "$PYTHON_PATH"
    log_success "Virtual environment ready"
}

install_deps() {
    cd "$INSTALL_DIR"
    export VIRTUAL_ENV="$INSTALL_DIR/venv"
    log_info "Installing Echo Agent dependencies..."
    if ! "$UV_CMD" pip install -e ".[all]"; then
        log_warn "Full install failed, falling back to base install."
        "$UV_CMD" pip install -e "."
    fi
    log_success "Dependencies installed"
}

get_command_link_dir() {
    echo "$HOME/.local/bin"
}

get_command_link_display_dir() {
    echo "~/.local/bin"
}

setup_path() {
    local echo_bin="$INSTALL_DIR/venv/bin/echo-agent"
    local link_dir
    local link_display_dir
    local original_path="$PATH"

    if [ ! -x "$echo_bin" ]; then
        log_error "echo-agent entry point not found at $echo_bin"
        exit 1
    fi

    link_dir="$(get_command_link_dir)"
    link_display_dir="$(get_command_link_display_dir)"
    mkdir -p "$link_dir"
    ln -sf "$echo_bin" "$link_dir/echo-agent"
    log_success "Symlinked echo-agent -> $link_display_dir/echo-agent"

    if echo "$original_path" | tr ':' '\n' | grep -qx "$link_dir"; then
        export PATH="$link_dir:$PATH"
        log_info "$link_display_dir already on PATH"
        return 0
    fi

    LOGIN_SHELL="$(basename "${SHELL:-/bin/bash}")"
    SHELL_CONFIGS=()
    IS_FISH=false

    case "$LOGIN_SHELL" in
        zsh)
            [ -f "$HOME/.zshrc" ] && SHELL_CONFIGS+=("$HOME/.zshrc")
            [ -f "$HOME/.zprofile" ] && SHELL_CONFIGS+=("$HOME/.zprofile")
            if [ ${#SHELL_CONFIGS[@]} -eq 0 ]; then
                touch "$HOME/.zshrc"
                SHELL_CONFIGS+=("$HOME/.zshrc")
            fi
            ;;
        bash)
            [ -f "$HOME/.bashrc" ] && SHELL_CONFIGS+=("$HOME/.bashrc")
            [ -f "$HOME/.bash_profile" ] && SHELL_CONFIGS+=("$HOME/.bash_profile")
            if [ ${#SHELL_CONFIGS[@]} -eq 0 ]; then
                touch "$HOME/.bashrc"
                SHELL_CONFIGS+=("$HOME/.bashrc")
            fi
            ;;
        fish)
            IS_FISH=true
            FISH_CONFIG="$HOME/.config/fish/config.fish"
            mkdir -p "$(dirname "$FISH_CONFIG")"
            touch "$FISH_CONFIG"
            ;;
        *)
            [ -f "$HOME/.bashrc" ] && SHELL_CONFIGS+=("$HOME/.bashrc")
            [ -f "$HOME/.zshrc" ] && SHELL_CONFIGS+=("$HOME/.zshrc")
            ;;
    esac

    PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
    for shell_config in "${SHELL_CONFIGS[@]}"; do
        if ! grep -v '^[[:space:]]*#' "$shell_config" 2>/dev/null | grep -q '\.local/bin'; then
            echo "" >> "$shell_config"
            echo "# Echo Agent" >> "$shell_config"
            echo "$PATH_LINE" >> "$shell_config"
            log_success "Added ~/.local/bin to PATH in $shell_config"
        fi
    done

    if [ "$IS_FISH" = true ]; then
        if ! grep -q 'fish_add_path.*\.local/bin' "$FISH_CONFIG" 2>/dev/null; then
            echo "" >> "$FISH_CONFIG"
            echo "# Echo Agent" >> "$FISH_CONFIG"
            echo 'fish_add_path "$HOME/.local/bin"' >> "$FISH_CONFIG"
            log_success "Added ~/.local/bin to PATH in $FISH_CONFIG"
        fi
    fi

    export PATH="$link_dir:$PATH"
}

prepare_home() {
    mkdir -p "$ECHO_HOME"
    log_success "Home directory ready: $ECHO_HOME"
}

run_setup_wizard() {
    local echo_cmd

    if [ "$RUN_SETUP" != true ]; then
        log_info "Skipping setup wizard (--skip-setup)"
        return 0
    fi

    echo_cmd="$INSTALL_DIR/venv/bin/echo-agent"
    if [ ! -x "$echo_cmd" ]; then
        return 0
    fi

    if prompt_yes_no "Run Echo Agent setup now?" "yes"; then
        "$echo_cmd" setup
    else
        log_info "You can run setup later with: echo-agent setup"
    fi
}

print_success() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│              Installation Complete                     │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}Paths:${NC}"
    echo "  Config:    $ECHO_HOME/echo-agent.yaml"
    echo "  Data:      $ECHO_HOME/data/"
    echo "  Code:      $INSTALL_DIR"
    echo ""
    echo -e "${CYAN}${BOLD}Commands:${NC}"
    echo "  echo-agent          Start CLI"
    echo "  echo-agent setup    Run setup wizard"
    echo "  echo-agent status   Show current config status"
    echo "  echo-agent gateway  Start gateway server"
    echo ""
    echo -e "${CYAN}${BOLD}If the command is not available yet:${NC}"
    case "$(basename "${SHELL:-/bin/bash}")" in
        zsh) echo "  source ~/.zshrc" ;;
        fish) echo "  source ~/.config/fish/config.fish" ;;
        *) echo "  source ~/.bashrc" ;;
    esac
    echo ""
    echo "To use a project-local workspace instead of ~/.echo-agent:"
    echo "  echo-agent setup -w /path/to/workspace"
}

main() {
    print_banner
    detect_os
    check_git
    install_uv
    check_python
    clone_repo
    setup_venv
    install_deps
    setup_path
    prepare_home
    run_setup_wizard
    print_success
}

main
