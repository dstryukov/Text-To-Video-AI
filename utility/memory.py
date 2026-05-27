import gc

def clear_system_memory():
    """Вызов Python garbage collector для очистки системной памяти."""
    gc.collect()

def clear_cuda_memory():
    """Очистка кэша аллокатора CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

def clear_memory():
    """Очистка всей доступной памяти (GC + CUDA)."""
    clear_system_memory()
    clear_cuda_memory()
    print("Memory cleared (System GC + CUDA Cache).")

def unload_model(model=None):
    """Выгрузка модели из видеопамяти и очистка кэша."""
    if model is not None:
        try:
            if hasattr(model, 'cpu'):
                model.cpu()
        except:
            pass
        del model
    clear_memory()
