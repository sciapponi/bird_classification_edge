#!/bin/bash

# =============================================================================
# UPLOAD PREPROCESSED DATASET SCRIPT
# =============================================================================
# Script per trasferire il dataset preprocessato via SCP a un server remoto
# Supporta compressione, resume, verifica integrità e progress monitoring
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURAZIONE DEFAULT
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_DATASET_PATH="$PROJECT_ROOT/bird_sound_dataset_processed"
DEFAULT_REMOTE_PATH="/data/datasets/"
COMPRESSION_ENABLED=true
VERIFY_TRANSFER=true
RESUME_ENABLED=true
PROGRESS_ENABLED=true

# =============================================================================
# COLORI PER OUTPUT
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# FUNZIONI UTILITY
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

format_size() {
    local size=$1
    if [ $size -lt 1024 ]; then
        echo "${size}B"
    elif [ $size -lt 1048576 ]; then
        echo "$(( size / 1024 ))KB"
    elif [ $size -lt 1073741824 ]; then
        echo "$(( size / 1048576 ))MB"
    else
        echo "$(( size / 1073741824 ))GB"
    fi
}

format_duration() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        echo "${seconds}s"
    elif [ $seconds -lt 3600 ]; then
        echo "$(( seconds / 60 ))m $(( seconds % 60 ))s"
    else
        echo "$(( seconds / 3600 ))h $(( (seconds % 3600) / 60 ))m"
    fi
}

check_requirements() {
    log_header "Verifica Requisiti"
    
    # Check commands
    local required_commands=("scp" "ssh" "rsync" "tar" "gzip")
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        log_error "Comandi mancanti: ${missing_commands[*]}"
        log_error "Installa i comandi richiesti e riprova"
        exit 1
    fi
    
    log_success "Tutti i requisiti soddisfatti"
}

test_ssh_connection() {
    local server="$1"
    local user="$2"
    
    log_info "Test connessione SSH a ${user}@${server}..."
    
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "${user}@${server}" 'exit' 2>/dev/null; then
        log_success "Connessione SSH funzionante"
        return 0
    else
        log_error "Impossibile connettersi a ${user}@${server}"
        log_error "Verifica:"
        echo "  - Server raggiungibile"
        echo "  - Credenziali SSH corrette"
        echo "  - Chiavi SSH configurate"
        return 1
    fi
}

get_dataset_info() {
    local dataset_path="$1"
    
    if [ ! -d "$dataset_path" ]; then
        log_error "Dataset non trovato: $dataset_path"
        exit 1
    fi
    
    log_header "Informazioni Dataset"
    
    local total_files=$(find "$dataset_path" -type f | wc -l)
    local total_size=$(du -sb "$dataset_path" 2>/dev/null | cut -f1 || echo "0")
    local species_count=$(find "$dataset_path" -maxdepth 1 -type d | wc -l)
    species_count=$((species_count - 1))  # Escludi directory root
    
    log_info "📁 Percorso: $dataset_path"
    log_info "📊 File totali: $total_files"
    log_info "📦 Dimensione totale: $(format_size $total_size)"
    log_info "🐦 Specie: $species_count"
    
    # Mostra alcune specie
    log_info "🔍 Specie presenti:"
    find "$dataset_path" -maxdepth 1 -type d -not -path "$dataset_path" | head -5 | while read -r dir; do
        local species=$(basename "$dir")
        local file_count=$(find "$dir" -type f | wc -l)
        echo "     • $species ($file_count file)"
    done
    
    if [ $species_count -gt 5 ]; then
        echo "     ... e altri $((species_count - 5)) specie"
    fi
    
    echo
    export DATASET_SIZE="$total_size"
    export DATASET_FILES="$total_files"
}

create_compressed_archive() {
    local dataset_path="$1"
    local archive_path="$2"
    
    log_header "Creazione Archivio Compresso"
    
    log_info "📦 Compressione in corso..."
    log_info "   Input: $dataset_path"
    log_info "   Output: $archive_path"
    
    local start_time=$(date +%s)
    
    # Usa tar con compressione gzip e progress
    cd "$(dirname "$dataset_path")"
    local dataset_name=$(basename "$dataset_path")
    
    if command -v pv &> /dev/null && [ "$PROGRESS_ENABLED" = true ]; then
        # Con progress bar se pv è disponibile
        tar -cf - "$dataset_name" | pv -s $(du -sb "$dataset_name" | cut -f1) | gzip > "$archive_path"
    else
        # Senza progress bar
        tar -czf "$archive_path" "$dataset_name"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local archive_size=$(stat -f%z "$archive_path" 2>/dev/null || stat -c%s "$archive_path" 2>/dev/null || echo "0")
    local compression_ratio=$((archive_size * 100 / DATASET_SIZE))
    
    log_success "📦 Archivio creato in $(format_duration $duration)"
    log_info "📊 Dimensione originale: $(format_size $DATASET_SIZE)"
    log_info "📊 Dimensione compressa: $(format_size $archive_size)"
    log_info "📊 Rapporto compressione: ${compression_ratio}%"
    
    export ARCHIVE_SIZE="$archive_size"
}

estimate_transfer_time() {
    local file_size="$1"
    local server="$2"
    local user="$3"
    
    log_header "Stima Tempo Trasferimento"
    
    log_info "🔍 Test velocità connessione..."
    
    # Crea un file di test piccolo
    local test_file="/tmp/speed_test_$(date +%s).dat"
    dd if=/dev/zero of="$test_file" bs=1024 count=1024 2>/dev/null  # 1MB
    
    local start_time=$(date +%s)
    if scp -o BatchMode=yes "$test_file" "${user}@${server}:/tmp/" 2>/dev/null; then
        local end_time=$(date +%s)
        local test_duration=$((end_time - start_time))
        
        if [ $test_duration -gt 0 ]; then
            local speed_bps=$((1048576 / test_duration))  # bytes per second
            local estimated_time=$((file_size / speed_bps))
            
            log_info "⚡ Velocità stimata: $(format_size $speed_bps)/s"
            log_info "⏱️  Tempo stimato: $(format_duration $estimated_time)"
            
            # Pulisci file di test
            ssh "${user}@${server}" "rm -f /tmp/$(basename "$test_file")" 2>/dev/null || true
        else
            log_warning "Test troppo veloce per stima accurata"
        fi
    else
        log_warning "Impossibile stimare velocità di trasferimento"
    fi
    
    rm -f "$test_file"
}

upload_with_scp() {
    local local_file="$1"
    local remote_server="$2"
    local remote_user="$3"
    local remote_path="$4"
    
    log_header "Upload via SCP"
    
    local remote_file="${remote_path}/$(basename "$local_file")"
    local scp_options=()
    
    # Opzioni SCP
    scp_options+=("-C")  # Compressione
    scp_options+=("-o" "BatchMode=yes")  # Non interattivo
    
    if [ "$PROGRESS_ENABLED" = true ]; then
        scp_options+=("-v")  # Verbose per progress
    fi
    
    log_info "🚀 Inizio trasferimento..."
    log_info "   File: $local_file"
    log_info "   Destinazione: ${remote_user}@${remote_server}:${remote_file}"
    
    local start_time=$(date +%s)
    
    if scp "${scp_options[@]}" "$local_file" "${remote_user}@${remote_server}:${remote_file}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local file_size=$(stat -f%z "$local_file" 2>/dev/null || stat -c%s "$local_file" 2>/dev/null || echo "0")
        local speed=$((file_size / duration))
        
        log_success "✅ Trasferimento completato in $(format_duration $duration)"
        log_info "📊 Velocità media: $(format_size $speed)/s"
        
        return 0
    else
        log_error "❌ Trasferimento fallito"
        return 1
    fi
}

upload_with_rsync() {
    local local_path="$1"
    local remote_server="$2"
    local remote_user="$3"
    local remote_path="$4"
    
    log_header "Upload via Rsync"
    
    local rsync_options=()
    rsync_options+=("-avz")  # Archive, verbose, compress
    rsync_options+=("--partial")  # Keep partial files
    
    if [ "$RESUME_ENABLED" = true ]; then
        rsync_options+=("--partial-dir=.rsync-partial")
    fi
    
    if [ "$PROGRESS_ENABLED" = true ]; then
        rsync_options+=("--progress")
    fi
    
    log_info "🚀 Inizio sincronizzazione..."
    log_info "   Sorgente: $local_path"
    log_info "   Destinazione: ${remote_user}@${remote_server}:${remote_path}"
    
    local start_time=$(date +%s)
    
    if rsync "${rsync_options[@]}" "$local_path/" "${remote_user}@${remote_server}:${remote_path}/"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_success "✅ Sincronizzazione completata in $(format_duration $duration)"
        return 0
    else
        log_error "❌ Sincronizzazione fallita"
        return 1
    fi
}

verify_transfer() {
    local local_file="$1"
    local remote_server="$2"
    local remote_user="$3"
    local remote_file="$4"
    
    log_header "Verifica Trasferimento"
    
    log_info "🔍 Verifica integrità file..."
    
    # Calcola checksum locale
    local local_checksum
    if command -v sha256sum &> /dev/null; then
        local_checksum=$(sha256sum "$local_file" | cut -d' ' -f1)
    elif command -v shasum &> /dev/null; then
        local_checksum=$(shasum -a 256 "$local_file" | cut -d' ' -f1)
    else
        log_warning "Impossibile calcolare checksum: comando non disponibile"
        return 0
    fi
    
    # Calcola checksum remoto
    local remote_checksum
    if command -v sha256sum &> /dev/null; then
        remote_checksum=$(ssh "${remote_user}@${remote_server}" "sha256sum '$remote_file'" 2>/dev/null | cut -d' ' -f1)
    elif ssh "${remote_user}@${remote_server}" "command -v shasum" &> /dev/null; then
        remote_checksum=$(ssh "${remote_user}@${remote_server}" "shasum -a 256 '$remote_file'" 2>/dev/null | cut -d' ' -f1)
    else
        log_warning "Impossibile calcolare checksum remoto"
        return 0
    fi
    
    if [ "$local_checksum" = "$remote_checksum" ]; then
        log_success "✅ Checksum verificato: file trasferito correttamente"
        log_info "   SHA256: $local_checksum"
        return 0
    else
        log_error "❌ Checksum non corrispondente!"
        log_error "   Locale:  $local_checksum"
        log_error "   Remoto:  $remote_checksum"
        return 1
    fi
}

cleanup() {
    log_header "Pulizia File Temporanei"
    
    # Rimuovi archivi temporanei se creati
    if [ -n "${TEMP_ARCHIVE:-}" ] && [ -f "$TEMP_ARCHIVE" ]; then
        log_info "🗑️  Rimozione archivio temporaneo: $TEMP_ARCHIVE"
        rm -f "$TEMP_ARCHIVE"
    fi
    
    log_success "Pulizia completata"
}

print_usage() {
    cat << EOF
🚀 UPLOAD PREPROCESSED DATASET SCRIPT

Trasferisce il dataset preprocessato a un server remoto via SCP/Rsync

USO:
    $0 [OPZIONI] <server> <user> [remote_path]

PARAMETRI:
    server          Indirizzo del server remoto (es: server.example.com)
    user           Username per la connessione SSH
    remote_path    Percorso remoto (default: $DEFAULT_REMOTE_PATH)

OPZIONI:
    -d, --dataset PATH      Percorso del dataset locale (default: dataset preprocessato)
    -c, --compress          Comprimi in archivio prima del trasferimento
    -nc, --no-compress      Non comprimere (trasferimento diretto)
    -r, --rsync            Usa rsync invece di scp
    -nv, --no-verify       Salta verifica integrità
    -np, --no-progress     Disabilita progress bar
    -nr, --no-resume       Disabilita resume per rsync
    -h, --help             Mostra questo aiuto

ESEMPI:
    # Upload compresso con SCP
    $0 myserver.com user123

    # Upload diretto con rsync
    $0 --rsync --no-compress myserver.com user123 /data/datasets/

    # Upload da dataset custom
    $0 -d /path/to/dataset myserver.com user123

NOTA:
    - Assicurati che le chiavi SSH siano configurate
    - Il server deve avere spazio sufficiente
    - La connessione deve essere stabile per file grandi

EOF
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

main() {
    local dataset_path="$DEFAULT_DATASET_PATH"
    local server=""
    local user=""
    local remote_path="$DEFAULT_REMOTE_PATH"
    local use_rsync=false
    local temp_archive=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dataset)
                dataset_path="$2"
                shift 2
                ;;
            -c|--compress)
                COMPRESSION_ENABLED=true
                shift
                ;;
            -nc|--no-compress)
                COMPRESSION_ENABLED=false
                shift
                ;;
            -r|--rsync)
                use_rsync=true
                shift
                ;;
            -nv|--no-verify)
                VERIFY_TRANSFER=false
                shift
                ;;
            -np|--no-progress)
                PROGRESS_ENABLED=false
                shift
                ;;
            -nr|--no-resume)
                RESUME_ENABLED=false
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            -*)
                log_error "Opzione sconosciuta: $1"
                print_usage
                exit 1
                ;;
            *)
                if [ -z "$server" ]; then
                    server="$1"
                elif [ -z "$user" ]; then
                    user="$1"
                elif [ -z "$remote_path" ] || [ "$remote_path" = "$DEFAULT_REMOTE_PATH" ]; then
                    remote_path="$1"
                else
                    log_error "Troppi argomenti posizionali"
                    print_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate required parameters
    if [ -z "$server" ] || [ -z "$user" ]; then
        log_error "Server e user sono obbligatori!"
        print_usage
        exit 1
    fi
    
    # Setup trap for cleanup
    trap cleanup EXIT
    
    log_header "🚀 UPLOAD DATASET PREPROCESSATO"
    log_info "📅 $(date)"
    log_info "🖥️  Server: $server"
    log_info "👤 User: $user"
    log_info "📁 Dataset: $dataset_path"
    log_info "🎯 Destinazione: $remote_path"
    
    # Run checks
    check_requirements
    get_dataset_info "$dataset_path"
    test_ssh_connection "$server" "$user"
    
    # Prepare file for transfer
    local file_to_transfer="$dataset_path"
    
    if [ "$COMPRESSION_ENABLED" = true ] && [ "$use_rsync" = false ]; then
        temp_archive="/tmp/$(basename "$dataset_path")_$(date +%Y%m%d_%H%M%S).tar.gz"
        create_compressed_archive "$dataset_path" "$temp_archive"
        file_to_transfer="$temp_archive"
        export TEMP_ARCHIVE="$temp_archive"
        
        # Estimate transfer time for compressed file
        estimate_transfer_time "$ARCHIVE_SIZE" "$server" "$user"
    else
        # Estimate transfer time for uncompressed dataset
        estimate_transfer_time "$DATASET_SIZE" "$server" "$user"
    fi
    
    # Transfer
    local transfer_success=false
    
    if [ "$use_rsync" = true ]; then
        if [ "$COMPRESSION_ENABLED" = true ]; then
            log_warning "Rsync con compressione non supportato, uso rsync diretto"
        fi
        if upload_with_rsync "$dataset_path" "$server" "$user" "$remote_path"; then
            transfer_success=true
        fi
    else
        if upload_with_scp "$file_to_transfer" "$server" "$user" "$remote_path"; then
            transfer_success=true
            
            # Verify transfer if enabled
            if [ "$VERIFY_TRANSFER" = true ]; then
                local remote_file="${remote_path}/$(basename "$file_to_transfer")"
                if ! verify_transfer "$file_to_transfer" "$server" "$user" "$remote_file"; then
                    transfer_success=false
                fi
            fi
        fi
    fi
    
    # Final status
    if [ "$transfer_success" = true ]; then
        log_header "🎉 TRASFERIMENTO COMPLETATO CON SUCCESSO"
        log_success "Dataset caricato su ${user}@${server}:${remote_path}"
        
        if [ "$COMPRESSION_ENABLED" = true ] && [ "$use_rsync" = false ]; then
            log_info "💡 Per estrarre l'archivio sul server:"
            echo "   ssh ${user}@${server} 'cd ${remote_path} && tar -xzf $(basename "$file_to_transfer")'"
        fi
        
        exit 0
    else
        log_header "❌ TRASFERIMENTO FALLITO"
        log_error "Verifica connessione e spazio su disco remoto"
        exit 1
    fi
}

# Run main function with all arguments
main "$@" 