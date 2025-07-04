from datasets import BirdSoundDataset, ESC50Dataset, download_and_extract_esc50
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from utils import check_model, check_forward_pass, count_precise_macs
import torch.nn as nn
from datasets.dataset_factory import create_no_birds_dataset

# TEST IMPORT DIRETTO
try:
    from models import Improved_Phi_GRU_ATT
    print("!!! TEST: Improved_Phi_GRU_ATT importata con successo da models !!!")
    test_model_params = {
        'num_classes': 2,
        'n_mel_bins': 64,
        'hidden_dim': 32,
        'n_fft': 400,
        'hop_length': 160,
        'matchbox': {'base_filters': 32} 
    }
    test_model = Improved_Phi_GRU_ATT(**test_model_params)
    print("!!! TEST: Improved_Phi_GRU_ATT istanziata con successo !!!")
    # print(test_model) # Commentato per brevità output
except ImportError as e:
    print(f"!!! TEST FALLITO: ImportError durante 'from models import Improved_Phi_GRU_ATT': {e} !!!")
    import sys
    print("DEBUG sys.path durante test import:", sys.path)
except Exception as e:
    print(f"!!! TEST FALLITO: Altra Eccezione durante test import/istanza: {e} !!!")
# FINE TEST IMPORT DIRETTO

import hydra 
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Aggiunto per verificare la presenza di actual_model in DataParallel/DDP
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import AdamW # Import AdamW directly

@hydra.main(version_base=None, config_path='./config', config_name='bird_classification')
def train(cfg: DictConfig):
    """
    Main training function for the bird classification model.
    
    Args:
        cfg: Hydra configuration
    """
    import torch
    
    # Force immediate CUDA initialization to prevent conflicts
    torch.cuda.is_available()

    # Set up logging
    log = logging.getLogger(__name__)
    experiment_name = cfg.experiment_name
    log.info(f"Experiment: {experiment_name}")
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Set random seeds for reproducibility
    seed = cfg.training.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Download datasets if needed
    if cfg.dataset.get("download_datasets", False):
        log.info("Checking for datasets...")
        if not os.path.exists(cfg.dataset.bird_data_dir):
            log.info("Bird dataset not found. Please download and organize bird sound data manually as specified in bird_data_dir.")
        
        # Download ESC-50 dataset
        if not os.path.exists(cfg.dataset.esc50_dir) or not any(os.scandir(cfg.dataset.esc50_dir)):
            log.info("ESC-50 dataset not found or empty. Downloading...")
            # Assumendo che download_and_extract_esc50 sia definito altrove o importato
            # from datasets import download_and_extract_esc50 
            try:
                from datasets import download_and_extract_esc50 # Assicurati che sia importabile
                esc50_dir_downloaded = download_and_extract_esc50()
                if esc50_dir_downloaded and os.path.exists(esc50_dir_downloaded):
                     cfg.dataset.esc50_dir = esc50_dir_downloaded
                else:
                    log.error(f"Failed to download or locate ESC-50 dataset. Please check download_and_extract_esc50 or download manually to {cfg.dataset.esc50_dir}.")
            except ImportError:
                log.error("datasets.download_and_extract_esc50 not found. Cannot download ESC-50 automatically.")
                log.error(f"Please download manually to {cfg.dataset.esc50_dir}.")

        else:
            log.info(f"ESC-50 dataset found at: {cfg.dataset.esc50_dir}")
    
    # --- Create Bird Sound Datasets Only First ---
    log.info("Creating initial bird sound datasets (train/val/test)...")
    
    # Retrieve split parameters from config
    validation_split = cfg.dataset.get('validation_split', 0.15)
    test_split = cfg.dataset.get('test_split', 0.15)
    split_seed = cfg.training.get('seed', 42) # Use the general training seed for splitting
    log.info(f"Using dataset split parameters: val={validation_split}, test={test_split}, seed={split_seed}")

    # Instantiate BirdSoundDataset for each subset
    # NOTE: Now create_combined_dataset is replaced by direct instantiation + later combining
    train_bird_dataset = instantiate(cfg.dataset.bird_dataset_config,
                                     root_dir=cfg.dataset.bird_data_dir,
                                     allowed_classes=list(cfg.dataset.allowed_bird_classes), # Assicurati sia una lista
                                     subset="training",
                                     validation_split=validation_split, 
                                     test_split=test_split,             
                                     split_seed=split_seed,             
                                     augment=cfg.dataset.augmentation.enabled,
                                     sr=cfg.dataset.sample_rate,
                                     clip_duration=cfg.dataset.clip_duration,
                                     extract_calls=cfg.dataset.get('extract_calls', True),
                                     background_dataset=None 
    )

    val_bird_dataset = instantiate(cfg.dataset.bird_dataset_config,
                                   root_dir=cfg.dataset.bird_data_dir,
                                   allowed_classes=list(cfg.dataset.allowed_bird_classes),
                                   subset="validation",
                                   validation_split=validation_split, 
                                   test_split=test_split,             
                                   split_seed=split_seed,             
                                   augment=False, 
                                   sr=cfg.dataset.sample_rate,
                                   clip_duration=cfg.dataset.clip_duration,
                                   extract_calls=cfg.dataset.get('extract_calls', True),
                                   background_dataset=None
    )

    test_bird_dataset = instantiate(cfg.dataset.bird_dataset_config,
                                    root_dir=cfg.dataset.bird_data_dir,
                                    allowed_classes=list(cfg.dataset.allowed_bird_classes),
                                    subset="testing",
                                    validation_split=validation_split, 
                                    test_split=test_split,             
                                    split_seed=split_seed,             
                                    augment=False, 
                                    sr=cfg.dataset.sample_rate,
                                    clip_duration=cfg.dataset.clip_duration,
                                    extract_calls=cfg.dataset.get('extract_calls', True),
                                    background_dataset=None
    )

    log.info(f"Initial bird samples: Train={len(train_bird_dataset)}, Val={len(val_bird_dataset)}, Test={len(test_bird_dataset)}")

    # --- Dynamically Calculate and Create "No Birds" Datasets ---
    log.info("Calculating target number of \'no birds\' samples...")

    num_bird_classes = train_bird_dataset.get_num_classes()
    no_birds_label = num_bird_classes 
    log.info(f"Number of bird classes: {num_bird_classes}. \'No Birds\' label index: {no_birds_label}")

    train_bird_counts = train_bird_dataset.get_class_counts()
    if not train_bird_counts:
        log.warning("Training bird dataset is empty! Cannot calculate average. Setting no_birds target to 0.")
        avg_train_bird_samples = 0
    else:
        avg_train_bird_samples = np.mean(list(train_bird_counts.values()))
        log.info(f"Average samples per bird class in training set: {avg_train_bird_samples:.2f}")

    target_num_no_birds_train = int(round(avg_train_bird_samples))
    log.info(f"Target \'no birds\' samples for training: {target_num_no_birds_train}")

    if len(train_bird_dataset) > 0:
        no_birds_ratio_train = target_num_no_birds_train / len(train_bird_dataset) if len(train_bird_dataset) > 0 else 0
        # Ensure validation and test "no_birds" are proportional to their bird sample sizes
        target_num_no_birds_val = int(round(len(val_bird_dataset) * no_birds_ratio_train))
        target_num_no_birds_test = int(round(len(test_bird_dataset) * no_birds_ratio_train))
    else:
        log.warning("Training bird dataset is empty! Setting val/test \'no birds\' target to 0.")
        target_num_no_birds_val = 0
        target_num_no_birds_test = 0
        
    log.info(f"Target \'no birds\' samples for validation: {target_num_no_birds_val}")
    log.info(f"Target \'no birds\' samples for testing: {target_num_no_birds_test}")

    # Create 'no birds' datasets
    # Assumendo che create_no_birds_dataset sia definito altrove o importato
    # from datasets.dataset_factory import create_no_birds_dataset
    try:
        from datasets.dataset_factory import create_no_birds_dataset # Assicurati che sia importabile
        train_no_birds_dataset = create_no_birds_dataset(
            num_samples=target_num_no_birds_train,
            no_birds_label=no_birds_label,
            esc50_dir=cfg.dataset.esc50_dir,
            bird_data_dir=cfg.dataset.bird_data_dir, # Può essere usato per escludere rumori di uccelli
            allowed_bird_classes=list(cfg.dataset.allowed_bird_classes),
            subset="training", # o un subset specifico di ESC50 per no_birds_train
            target_sr=cfg.dataset.sample_rate,
            clip_duration=cfg.dataset.clip_duration,
            esc50_no_bird_ratio=cfg.dataset.get('esc50_no_bird_ratio', 0.5),
            load_pregenerated=cfg.dataset.get('load_pregenerated_no_birds', False),
            pregenerated_dir=cfg.dataset.get('pregenerated_no_birds_dir', 'augmented_dataset/no_birds/')
        )
        
        val_no_birds_dataset = create_no_birds_dataset(
            num_samples=target_num_no_birds_val,
            no_birds_label=no_birds_label,
            esc50_dir=cfg.dataset.esc50_dir,
            bird_data_dir=cfg.dataset.bird_data_dir,
            allowed_bird_classes=list(cfg.dataset.allowed_bird_classes),
            subset="validation", # o un subset specifico di ESC50 per no_birds_val
            target_sr=cfg.dataset.sample_rate,
            clip_duration=cfg.dataset.clip_duration,
            esc50_no_bird_ratio=cfg.dataset.get('esc50_no_bird_ratio', 0.5),
            load_pregenerated=cfg.dataset.get('load_pregenerated_no_birds', False),
            pregenerated_dir=cfg.dataset.get('pregenerated_no_birds_dir', 'augmented_dataset/no_birds/')
        )

        test_no_birds_dataset = create_no_birds_dataset(
            num_samples=target_num_no_birds_test,
            no_birds_label=no_birds_label,
            esc50_dir=cfg.dataset.esc50_dir,
            bird_data_dir=cfg.dataset.bird_data_dir,
            allowed_bird_classes=list(cfg.dataset.allowed_bird_classes),
            subset="testing", # o un subset specifico di ESC50 per no_birds_test
            target_sr=cfg.dataset.sample_rate,
            clip_duration=cfg.dataset.clip_duration,
            esc50_no_bird_ratio=cfg.dataset.get('esc50_no_bird_ratio', 0.5),
            load_pregenerated=cfg.dataset.get('load_pregenerated_no_birds', False),
            pregenerated_dir=cfg.dataset.get('pregenerated_no_birds_dir', 'augmented_dataset/no_birds/')
        )
    except ImportError:
        log.error("datasets.dataset_factory.create_no_birds_dataset not found. Cannot create 'no_birds' datasets.")
        # Potresti voler uscire o continuare senza 'no_birds' a seconda della logica desiderata
        train_no_birds_dataset, val_no_birds_dataset, test_no_birds_dataset = None, None, None


    # --- Combine Bird and "No Birds" Datasets ---
    log.info("Combining bird and \'no birds\' datasets...")
    if train_no_birds_dataset and len(train_no_birds_dataset) > 0:
        full_train_dataset = ConcatDataset([train_bird_dataset, train_no_birds_dataset])
    else:
        log.warning("Train 'no_birds' dataset is empty or not created. Using only bird sounds for training.")
        full_train_dataset = train_bird_dataset
        
    if val_no_birds_dataset and len(val_no_birds_dataset) > 0:
        full_val_dataset = ConcatDataset([val_bird_dataset, val_no_birds_dataset])
    else:
        log.warning("Validation 'no_birds' dataset is empty or not created. Using only bird sounds for validation.")
        full_val_dataset = val_bird_dataset

    if test_no_birds_dataset and len(test_no_birds_dataset) > 0:
        full_test_dataset = ConcatDataset([test_bird_dataset, test_no_birds_dataset])
    else:
        log.warning("Test 'no_birds' dataset is empty or not created. Using only bird sounds for testing.")
        full_test_dataset = test_bird_dataset
        
    log.info(f"Final training samples: {len(full_train_dataset)}")
    log.info(f"Final validation samples: {len(full_val_dataset)}")
    log.info(f"Final testing samples: {len(full_test_dataset)}")

    # Dataloaders
    train_loader = DataLoader(full_train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(full_val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(full_test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate model, optimizer, scheduler
    # Aggiorna num_classes nel modello per includere la classe 'no_birds'
    actual_num_classes = num_bird_classes + 1 if (train_no_birds_dataset and len(train_no_birds_dataset) > 0) else num_bird_classes
    
    # Prepara gli argomenti di override per l'istanziazione del modello
    model_instantiate_overrides = {'num_classes': actual_num_classes}
    
    # Passa f_min and f_max from dataset config to model if they are present and not None
    if hasattr(cfg.dataset, 'lowcut') and cfg.dataset.lowcut is not None:
        model_instantiate_overrides['f_min'] = cfg.dataset.lowcut
    if hasattr(cfg.dataset, 'highcut') and cfg.dataset.highcut is not None:
        model_instantiate_overrides['f_max'] = cfg.dataset.highcut

    # Istanzia il modello usando cfg.model e applicando gli overrides
    # cfg.model contiene già gli altri parametri statici del modello.
    model = instantiate(cfg.model, **model_instantiate_overrides).to(device)
    
    # Get learning rates and other optimizer hyperparams as Python primitives
    main_lr = float(cfg.optimizer.lr)
    # Recupera i learning rate specifici per breakpoint e transition_width
    breakpoint_lr_config = cfg.optimizer.get('breakpoint_lr')
    transition_width_lr_config = cfg.optimizer.get('transition_width_lr')
    
    # Usa main_lr come fallback se i LR specifici non sono impostati o sono None
    bp_lr = float(breakpoint_lr_config) if breakpoint_lr_config is not None else main_lr
    tw_lr = float(transition_width_lr_config) if transition_width_lr_config is not None else main_lr
    
    weight_decay = float(cfg.optimizer.get('weight_decay', 0.0)) # Default to 0.0 if not present

    optimizer_params_list = []

    if cfg.model.spectrogram_type == "combined_log_linear" and hasattr(model, 'combined_log_linear_spec'):
        # Nomi dei parametri specifici da cercare
        breakpoint_param_name = 'combined_log_linear_spec.breakpoint'
        transition_width_param_name = 'combined_log_linear_spec.transition_width'
        
        current_breakpoint_params = []
        current_transition_width_params = []
        current_other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if breakpoint_param_name in name:
                current_breakpoint_params.append(param)
            elif transition_width_param_name in name:
                current_transition_width_params.append(param)
            else:
                current_other_params.append(param)
        
        # Aggiungi gruppo per gli altri parametri (se presenti)
        if current_other_params:
             optimizer_params_list.append({'params': current_other_params, 'lr': main_lr})
        
        # Aggiungi gruppo per i parametri del breakpoint (se presenti)
        if current_breakpoint_params:
            optimizer_params_list.append({'params': current_breakpoint_params, 'lr': bp_lr})
            log.info(f"Using specific LR for breakpoint parameters: {bp_lr}.")
        else:
            log.info("No trainable 'breakpoint' parameters found or 'breakpoint_lr' not used.")

        # Aggiungi gruppo per i parametri della transition_width (se presenti)
        if current_transition_width_params:
            optimizer_params_list.append({'params': current_transition_width_params, 'lr': tw_lr})
            log.info(f"Using specific LR for transition_width parameters: {tw_lr}.")
        else:
            log.info("No trainable 'transition_width' parameters found or 'transition_width_lr' not used.")

        if not current_other_params and not current_breakpoint_params and not current_transition_width_params:
             log.warning("combined_log_linear_spec specified, but no trainable parameters found at all. Using main_lr for all parameters if any exist later.")
             # Fallback se nessun parametro è stato trovato, anche se è improbabile se il modello ha combined_log_linear_spec
             all_trainable_params = [p for p in model.parameters() if p.requires_grad]
             if all_trainable_params:
                  optimizer_params_list = [{'params': all_trainable_params, 'lr': main_lr}]

        # Check if all params were sorted into a group
        num_total_model_params = sum(1 for p in model.parameters() if p.requires_grad)
        num_grouped_params = len(current_other_params) + len(current_breakpoint_params) + len(current_transition_width_params)
        if num_total_model_params != num_grouped_params and (current_breakpoint_params or current_transition_width_params or current_other_params) : # check only if some groups were populated
            log.warning(f"Parameter count mismatch: Total trainable = {num_total_model_params}, Grouped = {num_grouped_params}. Some params might be missed!")
            # If mismatch, and specific LRs were intended, this could be an issue.
            # Ensure all parameters intended for specific LRs are correctly named.

    else:
        # Default: all parameters in one group with main_lr
        all_trainable_params = [p for p in model.parameters() if p.requires_grad]
        if all_trainable_params:
            optimizer_params_list.append({'params': all_trainable_params, 'lr': main_lr})

    if not optimizer_params_list:
        log.error("No trainable parameters found for the optimizer. Check model parameters and requires_grad settings.")
        # Handle this case, e.g., by raising an error or exiting, as AdamW needs params.
        # For now, we'll let it potentially fail in AdamW constructor if list is empty.
        # Or, provide a dummy parameter if that's a valid way to handle it (unlikely for real training)
        # For example, if the model truly has no trainable params, this list would be empty.
        # AdamW([]) would raise ValueError:optimizer got an empty parameter list
        # So, if optimizer_params_list is empty, we might need to skip optimizer creation or raise a clearer error.
        # However, a model should typically have trainable parameters.
        if not list(model.parameters()): # Check if model has any parameters at all
            log.warning("Model has no parameters at all.")
            # If the model is empty, then optimizer_params_list will be empty.
            # AdamW constructor will fail. This indicates a problem with the model itself.
        else: # Model has parameters, but none are trainable / not added to groups
             log.error("Model has parameters, but none were added to optimizer groups (e.g. all require_grad=False or logic error).")
        # Let's ensure AdamW gets a non-empty list if there are params, otherwise it's a model issue.
        # If optimizer_params_list is truly empty due to no trainable params, AdamW will raise an error.

    # Instantiate AdamW directly
    # AdamW requires a learning rate at the top level, even if groups override it.
    # This top-level LR acts as a default. We'll use main_lr.
    optimizer = AdamW(optimizer_params_list, lr=main_lr, weight_decay=weight_decay)
    
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    criterion = nn.CrossEntropyLoss()

    # Log model architecture and parameter count
    log.info("Model architecture:")
    log.info(str(model))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")
    
    # Check model and forward pass (optional, for debugging)
    # input_shape = (cfg.training.batch_size, int(cfg.dataset.sample_rate * cfg.dataset.clip_duration)) # Waveform input
    # check_model(model, input_shape, device, log)
    # check_forward_pass(model, input_shape, device, log)

    # Compute MACs (optional, for model complexity analysis)
    try:
        # Input for MACs computation needs to be a single sample, not a batch for ptflops
        # The shape depends on whether the model expects raw audio or spectrogram
        # If model.spectrogram_type suggests the model handles raw audio internally:
        macs_input_shape = (1, int(cfg.dataset.sample_rate * cfg.dataset.clip_duration))
        # If the model expects spectrograms (e.g., if you change dataset to output spectrograms):
        # n_feat = model.n_mel_bins if cfg.model.spectrogram_type == 'mel' else cfg.model.n_linear_filters # or n_fft // 2 + 1
        # time_steps = int((cfg.dataset.sample_rate * cfg.dataset.clip_duration - cfg.model.n_fft) / cfg.model.hop_length + 1)
        # macs_input_shape = (1, n_feat, time_steps) # Spectrogram input for ptflops

        log.info("Computing model complexity...")
        # Note: count_precise_macs expects input_constructor that yields a tensor of the correct shape
        # It does not directly take the shape tuple for ptflops
        macs_input_constructor = lambda: torch.randn(macs_input_shape).to(device)

        macs_str = count_precise_macs(model, macs_input_constructor)
        log.info(f"Model MACs: {macs_str}")
    except Exception as e:
        log.warning(f"Could not compute MACs: {e}")


    # Training loop
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'lr': []
    }
    # Aggiungi chiavi per breakpoint e transition_width se si usa combined_log_linear
    if cfg.model.spectrogram_type == "combined_log_linear":
        history['breakpoint'] = []
        history['transition_width'] = []

    log.info("Starting training...")
    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]", unit="batch")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            train_pbar.set_postfix(loss=loss.item(), acc=100. * correct_predictions / total_predictions if total_predictions > 0 else 0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_predictions / total_predictions
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val_predictions = 0
        total_val_predictions = 0
        all_val_labels = []       # Per metriche aggiuntive
        all_val_predictions = []  # Per metriche aggiuntive
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Val]", unit="batch")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_predictions += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()
                val_pbar.set_postfix(loss=loss.item(), acc=100. * correct_val_predictions / total_val_predictions if total_val_predictions > 0 else 0)

                all_val_labels.extend(labels.cpu().numpy())             # Raccogli etichette
                all_val_predictions.extend(predicted.cpu().numpy())   # Raccogli predizioni

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val_predictions / total_val_predictions
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Calcola e logga Precision, Recall, F1 per la validazione
        # Utilizza le etichette definite per il test set se disponibili, altrimenti range
        # Questo è importante per classification_report, meno per le metriche aggregate
        val_class_labels_for_report = class_labels if 'class_labels' in locals() and class_labels else list(range(actual_num_classes))

        precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(
            all_val_labels, 
            all_val_predictions, 
            average='weighted', 
            labels=np.unique(all_val_labels + all_val_predictions), # Considera tutte le etichette presenti
            zero_division=0
        )
        history['val_precision'].append(precision_val)
        history['val_recall'].append(recall_val)
        history['val_f1'].append(f1_val)

        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Log dei parametri del filtro differenziabile
        if cfg.model.spectrogram_type == "combined_log_linear":
            actual_model = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
            if hasattr(actual_model, 'combined_log_linear_spec') and actual_model.combined_log_linear_spec is not None:
                current_breakpoint = actual_model.combined_log_linear_spec.breakpoint.item()
                current_transition_width = actual_model.combined_log_linear_spec.transition_width.item()
                history['breakpoint'].append(current_breakpoint)
                history['transition_width'].append(current_transition_width)
                log.info(f"Epoch {epoch+1} - Breakpoint: {current_breakpoint:.2f} Hz, Transition Width: {current_transition_width:.2f}")
            else: # Caso in cui combined_log_linear_spec non è stato inizializzato come atteso
                history['breakpoint'].append(float('nan'))
                history['transition_width'].append(float('nan'))
                log.warning(f"Epoch {epoch+1} - combined_log_linear_spec not found on model, though spectrogram_type is {cfg.model.spectrogram_type}")
        # Aggiungi placeholder se le chiavi esistono ma non siamo in combined_log_linear per questa epoca (improbabile se cfg non cambia)
        elif 'breakpoint' in history and len(history['breakpoint']) <= epoch : 
             history['breakpoint'].append(float('nan'))
             history['transition_width'].append(float('nan'))


        log.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        log.info(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}%")
        log.info(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
        log.info(f"Val Precision (w): {precision_val:.4f}, Val Recall (w): {recall_val:.4f}, Val F1 (w): {f1_val:.4f}")
        log.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping and model saving
        if epoch_val_loss < best_val_loss - cfg.training.min_delta:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc # Aggiorna anche la best_val_acc quando la loss migliora
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"{experiment_name}_best_model.pth"))
            log.info(f"Validation loss improved from {best_val_loss + cfg.training.min_delta:.4f} to {best_val_loss:.4f}")
            log.info(f"New best model saved with val acc: {best_val_acc*100:.2f}%")
        else: # O se la val_acc è migliorata, anche se la loss non strettamente
            if epoch_val_acc > best_val_acc : # Salva anche se solo l'accuracy migliora (comune in classificazione)
                 best_val_acc = epoch_val_acc
                 # Non resettare patience_counter qui a meno che non sia la metrica primaria di early stopping
                 torch.save(model.state_dict(), os.path.join(output_dir, f"{experiment_name}_best_model_acc.pth")) # Nome diverso per chiarezza
                 log.info(f"New best model saved with val acc: {best_val_acc*100:.2f}% (val_loss: {epoch_val_loss:.4f})")
            
            patience_counter += 1
            log.info(f"No improvement in validation loss for {patience_counter}/{cfg.training.patience} epochs")

        if patience_counter >= cfg.training.patience:
            log.info("Early stopping triggered.")
            break
        
        scheduler.step(epoch_val_loss) # Passa la validation loss allo scheduler

    # Plot training history
    # Assumendo che plot_training_history sia definito altrove o importato
    # from utils import plot_training_history
    try:
        plot_training_history(history, output_dir, experiment_name, cfg.training.epochs) # Passa num_epochs
    except NameError:
        log.error("Function 'plot_training_history' not found. Cannot generate training history plot.")
    except Exception as e:
        log.error(f"Error plotting training history: {e}")


    # Test
    log.info("\nEvaluating on test set...")
    if os.path.exists(os.path.join(output_dir, f"{experiment_name}_best_model.pth")):
        model.load_state_dict(torch.load(os.path.join(output_dir, f"{experiment_name}_best_model.pth")))
        log.info("Loaded best model based on validation loss for testing.")
    elif os.path.exists(os.path.join(output_dir, f"{experiment_name}_best_model_acc.pth")): # Fallback al modello con best accuracy
        model.load_state_dict(torch.load(os.path.join(output_dir, f"{experiment_name}_best_model_acc.pth")))
        log.info("Loaded best model based on validation accuracy for testing.")
    else:
        log.warning("No best model found to load for testing. Using the model from the last epoch.")
        
    model.eval()
    test_loss = 0.0
    correct_test_predictions = 0
    total_test_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test_predictions += labels.size(0)
            correct_test_predictions += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    final_test_loss = test_loss / len(test_loader.dataset)
    final_test_acc = correct_test_predictions / total_test_predictions
    log.info(f"Test Loss: {final_test_loss:.4f}, Test Acc: {final_test_acc*100:.2f}%")

    # Determina le etichette delle classi per il report e la confusion matrix
    # La fonte di verità per il numero di classi è actual_num_classes.
    # La fonte di verità per i nomi delle classi di uccelli è cfg.dataset.allowed_bird_classes.

    temp_class_names = list(cfg.dataset.allowed_bird_classes)
    has_no_bird_class = (train_no_birds_dataset and len(train_no_birds_dataset) > 0)

    if actual_num_classes == len(temp_class_names) + 1 and has_no_bird_class:
        # Caso corretto: numero di classi del modello = uccelli + "non-bird"
        report_target_names = temp_class_names + ["non-bird"]
    elif actual_num_classes == len(temp_class_names) and not has_no_bird_class:
        # Caso corretto: numero di classi del modello = solo uccelli
        report_target_names = temp_class_names
    else:
        # C'è un disallineamento. Loggalo e usa etichette numeriche come fallback.
        log.warning(
            f"Mismatch in class name definition. "
            f"actual_num_classes: {actual_num_classes}, "
            f"len(cfg.dataset.allowed_bird_classes): {len(temp_class_names)}, "
            f"has_no_bird_class: {has_no_bird_class}. "
            "Using numeric labels for classification report."
        )
        report_target_names = [str(i) for i in range(actual_num_classes)]

    # Verifica finale della lunghezza (dovrebbe sempre corrispondere ora, a meno del fallback)
    if len(report_target_names) != actual_num_classes:
        log.error(
            f"CRITICAL: Final length of report_target_names ({len(report_target_names)}) "
            f"does not match actual_num_classes ({actual_num_classes}) even after logic adjustment. "
            "Defaulting to numeric labels."
        )
        report_target_names = [str(i) for i in range(actual_num_classes)]

    # Calcola e logga Precision, Recall, F1 per il test set (aggregate)
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='weighted', 
        labels=np.unique(all_labels + all_predictions), # Considera tutte le etichette presenti nel test set
        zero_division=0
    )
    log.info(f"Test Precision (w): {precision_test:.4f}, Test Recall (w): {recall_test:.4f}, Test F1 (w): {f1_test:.4f}")

    # Genera e logga il classification report per classe
    # Ora usiamo 'report_target_names' che dovrebbe contenere i nomi corretti o un fallback sicuro.
    try:
        test_classification_rep = classification_report(
            all_labels, 
            all_predictions, 
            target_names=report_target_names, # USA I NOMI DELLE CLASSI DEFINITI SOPRA
            labels=np.arange(actual_num_classes), # Usa tutte le possibili etichette da 0 a N-1 classi
            zero_division=0
        )
        log.info(f"Classification Report (Test Set):\n{test_classification_rep}")
    except Exception as e:
        log.error(f"Could not generate classification report: {e}")
        test_classification_rep = "Error generating report."

    # Save confusion matrix
    try:
        # Passa report_target_names anche a save_confusion_matrix per coerenza
        save_confusion_matrix(all_labels, all_predictions, report_target_names, output_dir)
    except NameError:
        log.error("Function 'save_confusion_matrix' not found. Cannot generate confusion matrix plot.")
    except Exception as e:
        log.error(f"Error generating confusion matrix: {e}")


    # Save results
    results = {
        "experiment_name": experiment_name,
        "best_val_acc": best_val_acc, # Questa è la best_val_acc corrispondente a best_val_loss o la best_acc assoluta
        "test_acc": final_test_acc,
        "test_precision_weighted": precision_test, # NUOVA METRICA
        "test_recall_weighted": recall_test,     # NUOVA METRICA
        "test_f1_weighted": f1_test,             # NUOVA METRICA
        "total_params": total_params,
        # "macs": macs_str if 'macs_str' in locals() else "Not computed" # Aggiungi se calcolato
    }
    if 'macs_str' in locals(): # Per evitare NameError se il calcolo MACs fallisce
        results["macs"] = macs_str
    else:
        results["macs"] = "Not computed"


    # Save results to JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Save model summary (compresi i parametri del filtro se presenti alla fine del training)
    summary_path = os.path.join(output_dir, "model_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Model: {cfg.model._target_}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        if 'macs_str' in locals():
             f.write(f"MACs: {macs_str}\n")
        else:
             f.write("MACs: Not computed\n")

        if cfg.model.spectrogram_type == "combined_log_linear":
            actual_model = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
            if hasattr(actual_model, 'combined_log_linear_spec') and actual_model.combined_log_linear_spec is not None:
                final_breakpoint = actual_model.combined_log_linear_spec.breakpoint.item()
                final_transition_width = actual_model.combined_log_linear_spec.transition_width.item()
                f.write(f"Final Differentiable Filter Breakpoint: {final_breakpoint:.2f} Hz\n")
                f.write(f"Final Differentiable Filter Transition Width: {final_transition_width:.2f}\n")
        f.write(f"\nBest validation accuracy: {best_val_acc*100:.2f}%\n") # Usa la best_val_acc registrata
        f.write(f"Test accuracy: {final_test_acc*100:.2f}%\n")
        f.write(f"Test Precision (weighted): {precision_test:.4f}\n")
        f.write(f"Test Recall (weighted): {recall_test:.4f}\n")
        f.write(f"Test F1 (weighted): {f1_test:.4f}\n")
        f.write(f"\nClassification Report (Test Set):\n{test_classification_rep}\n") # AGGIUNTO REPORT
    log.info(f"Model summary saved to {summary_path}")
    log.info(f"Training completed! Best validation accuracy: {best_val_acc*100:.2f}%")
    log.info(f"Test accuracy: {final_test_acc*100:.2f}%")
    log.info(f"Output directory: {output_dir}")


# Funzione per plottare la history (spostata o definita qui)
def plot_training_history(history, output_dir, experiment_name, num_epochs_config):
    log = logging.getLogger(__name__) 
    
    actual_epochs_run = len(history['train_loss'])

    # Determina il numero di subplot necessari
    num_metric_plots = 2 # Loss e Accuracy base
    plot_filter_params = False
    plot_lr = False
    plot_val_prf = False # Flag per P/R/F1

    if 'breakpoint' in history and 'transition_width' in history:
        if len(history['breakpoint']) == actual_epochs_run and len(history['transition_width']) == actual_epochs_run:
            if not all(np.isnan(history['breakpoint'])) and not all(np.isnan(history['transition_width'])):
                plot_filter_params = True
                num_metric_plots += 2 
        else:
            log.warning("Breakpoint/Transition width history length mismatch. Skipping their plot.")

    if 'lr' in history and len(history['lr']) == actual_epochs_run:
        plot_lr = True
        num_metric_plots +=1
    
    # Controlla per Precision, Recall, F1 di validazione
    if 'val_precision' in history and 'val_recall' in history and 'val_f1' in history:
        if (len(history['val_precision']) == actual_epochs_run and 
            len(history['val_recall']) == actual_epochs_run and 
            len(history['val_f1']) == actual_epochs_run):
            # Ulteriore controllo per assicurarsi che non siano tutti NaN se dovessero esserlo per qualche motivo
            if not (all(np.isnan(history['val_precision'])) and 
                    all(np.isnan(history['val_recall'])) and 
                    all(np.isnan(history['val_f1']))):
                plot_val_prf = True
                num_metric_plots += 3 # Un plot per Precision, uno per Recall, uno per F1
        else:
            log.warning("Validation P/R/F1 history length mismatch or missing. Skipping their plots.")

    fig, axs = plt.subplots(num_metric_plots, 1, figsize=(12, num_metric_plots * 4), sharex=True) # sharex=True
    if num_metric_plots == 1 and not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    
    epochs_range = range(1, actual_epochs_run + 1)
    current_ax_idx = 0

    # Plot Loss
    axs[current_ax_idx].plot(epochs_range, history['train_loss'], label='Train Loss', marker='.', linestyle='-')
    axs[current_ax_idx].plot(epochs_range, history['val_loss'], label='Val Loss', marker='.', linestyle='-')
    axs[current_ax_idx].set_title(f'{experiment_name} - Loss per Epoch')
    axs[current_ax_idx].set_ylabel('Loss')
    axs[current_ax_idx].legend()
    axs[current_ax_idx].grid(True)
    current_ax_idx += 1

    # Plot Accuracy
    axs[current_ax_idx].plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='.', linestyle='-')
    axs[current_ax_idx].plot(epochs_range, history['val_acc'], label='Val Accuracy', marker='.', linestyle='-')
    axs[current_ax_idx].set_title(f'{experiment_name} - Accuracy per Epoch')
    axs[current_ax_idx].set_ylabel('Accuracy')
    axs[current_ax_idx].legend()
    axs[current_ax_idx].grid(True)
    current_ax_idx += 1

    if plot_val_prf:
        # Plot Validation Precision
        axs[current_ax_idx].plot(epochs_range, history['val_precision'], label='Val Precision (w)', color='cyan', marker='.', linestyle='-')
        axs[current_ax_idx].set_title(f'{experiment_name} - Weighted Val Precision per Epoch')
        axs[current_ax_idx].set_ylabel('Precision (w)')
        axs[current_ax_idx].legend()
        axs[current_ax_idx].grid(True)
        current_ax_idx += 1

        # Plot Validation Recall
        axs[current_ax_idx].plot(epochs_range, history['val_recall'], label='Val Recall (w)', color='magenta', marker='.', linestyle='-')
        axs[current_ax_idx].set_title(f'{experiment_name} - Weighted Val Recall per Epoch')
        axs[current_ax_idx].set_ylabel('Recall (w)')
        axs[current_ax_idx].legend()
        axs[current_ax_idx].grid(True)
        current_ax_idx += 1

        # Plot Validation F1-score
        axs[current_ax_idx].plot(epochs_range, history['val_f1'], label='Val F1-score (w)', color='brown', marker='.', linestyle='-')
        axs[current_ax_idx].set_title(f'{experiment_name} - Weighted Val F1-score per Epoch')
        axs[current_ax_idx].set_ylabel('F1-score (w)')
        axs[current_ax_idx].legend()
        axs[current_ax_idx].grid(True)
        current_ax_idx += 1

    if plot_filter_params:
        # Plot Breakpoint
        axs[current_ax_idx].plot(epochs_range, history['breakpoint'], label='Breakpoint (Hz)', color='green', marker='.', linestyle='-')
        axs[current_ax_idx].set_title(f'{experiment_name} - Differentiable Filter Breakpoint per Epoch')
        axs[current_ax_idx].set_ylabel('Breakpoint (Hz)')
        axs[current_ax_idx].legend()
        axs[current_ax_idx].grid(True)
        current_ax_idx += 1

        # Plot Transition Width
        axs[current_ax_idx].plot(epochs_range, history['transition_width'], label='Transition Width', color='purple', marker='.', linestyle='-')
        axs[current_ax_idx].set_title(f'{experiment_name} - Differentiable Filter Transition Width per Epoch')
        axs[current_ax_idx].set_ylabel('Transition Width')
        axs[current_ax_idx].legend()
        axs[current_ax_idx].grid(True)
        current_ax_idx += 1
    
    if plot_lr:
        axs[current_ax_idx].plot(epochs_range, history['lr'], label='Learning Rate', color='red', marker='.', linestyle='-')
        axs[current_ax_idx].set_title(f'{experiment_name} - Learning Rate per Epoch')
        axs[current_ax_idx].set_xlabel('Epoch') # xlabel solo per l'ultimo subplot se sharex=True
        axs[current_ax_idx].set_ylabel('Learning Rate')
        axs[current_ax_idx].legend()
        axs[current_ax_idx].grid(True)
        # current_ax_idx += 1 # Non necessario
    
    # Imposta xticks per tutti i subplot se sharex=True, o per l'ultimo se non lo è e non è l'ultimo asse.
    # Se sharex=True, basta impostarlo per l'ultimo asse visibile.
    for ax in axs: # Assicura che tutti gli assi abbiano i tick delle epoche corretti
        ax.set_xticks(epochs_range)
        # Imposta xlim per evitare spazi vuoti se actual_epochs_run è piccolo
        if actual_epochs_run > 0:
            ax.set_xlim(0.5, actual_epochs_run + 0.5)

    if not plot_lr: # Se LR non è l'ultimo plot, aggiungi xlabel all'ultimo plot corrente
        axs[current_ax_idx -1].set_xlabel('Epoch')


    plt.tight_layout(pad=2.0) # Riduci padding se molti subplot
    plot_path = os.path.join(output_dir, "training_history.png")
    try:
        plt.savefig(plot_path)
        log.info(f"Training history plot saved to {plot_path}")
    except Exception as e:
        log.error(f"Failed to save training history plot: {e}")
    finally:
        plt.close()

def save_confusion_matrix(y_true, y_pred, classes, output_dir):
    log = logging.getLogger(__name__)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    try:
        cm_df.to_csv(csv_path)
        log.info(f"Confusion matrix CSV saved to {csv_path}")
    except Exception as e:
        log.error(f"Failed to save confusion matrix CSV: {e}")

    # Save PNG
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    png_path = os.path.join(output_dir, "confusion_matrix.png")
    try:
        plt.savefig(png_path)
        log.info(f"Confusion matrix PNG saved to {png_path}")
    except Exception as e:
        log.error(f"Failed to save confusion matrix PNG: {e}")
    finally:
        plt.close()


if __name__ == '__main__':
    # Import necessari per il funzionamento standalone dello script,
    # se non sono già importati globalmente nel file.
    # Questi import sono necessari se si esegue lo script direttamente
    # e non sono già coperti dagli import all'inizio del file.
    from datasets import BirdSoundDataset, ESC50Dataset, download_and_extract_esc50
    from torch.utils.data import DataLoader, ConcatDataset
    from tqdm import tqdm
    from utils import check_model, check_forward_pass, count_precise_macs
    import torch.nn as nn
    from datasets.dataset_factory import create_no_birds_dataset
    
    # Questo try-except blocco è per testare l'importazione del modello
    # e potrebbe non essere necessario per il funzionamento normale.
    try:
        from models import Improved_Phi_GRU_ATT
        print("!!! TEST (main): Improved_Phi_GRU_ATT importata con successo da models !!!")
    except ImportError as e:
        print(f"!!! TEST FALLITO (main): ImportError durante \'from models import Improved_Phi_GRU_ATT\': {e} !!!")
        import sys
        print("DEBUG sys.path durante test import (main):", sys.path)
    except Exception as e:
        print(f"!!! TEST FALLITO (main): Altra Eccezione durante test import/istanza: {e} !!!")

    train()