@echo off
setlocal enabledelayedexpansion
REM Exit on any error
set "EXIT_ON_ERROR=1"

REM Always run from the project root
cd /d "%~dp0"

REM Environment variables for TensorFlow settings
set TF_GPU_ALLOCATOR=cuda_malloc
set XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
set TF_CUDNN_WORKSPACE_LIMIT_IN_MB=64
set TF_CUDNN_USE_AUTOTUNE=0
set TF_DETERMINISTIC_OPS=1
set TF_XLA_FLAGS=--tf_xla_auto_jit=0

REM Activate virtual environment and run Python training
call venv\Scripts\activate
python cldice_model.py --text_dir train_images_augmented --bars_dir train_bars_augmented

if errorlevel 1 (
    echo Training script failed.
    exit /b 1
)
echo Training completed successfully.
