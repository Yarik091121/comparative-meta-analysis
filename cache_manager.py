# cache_manager.py
"""
Менеджер кэширования на основе diskcache.
"""

import os
import pickle
import diskcache
from config import CACHE_DIR


class CacheManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = diskcache.Cache(CACHE_DIR)
        return cls._instance
    
    @property
    def cache(self):
        return self._instance._cache if hasattr(self, '_instance') else self._cache
    
    def get(self, key: str):
        return self._cache.get(key, default=None)
    
    def set(self, key: str, value, expire: int = 86400):
        self._cache.set(key, value, expire=expire)
    
    def has(self, key: str) -> bool:
        return key in self._cache
    
    def delete(self, key: str):
        if key in self._cache:
            self._cache.delete(key)
    
    def clear_all(self):
        self._cache.clear()
    
    def save_dataframe(self, key: str, df):
        """Сохраняет DataFrame через pickle."""
        self.set(f"df_{key}", pickle.dumps(df))
    
    def load_dataframe(self, key: str):
        """Загружает DataFrame из кэша."""
        data = self.get(f"df_{key}")
        if data is not None:
            return pickle.loads(data)
        return None
    
    def save_model(self, key: str, model):
        self.set(f"model_{key}", pickle.dumps(model))
    
    def load_model(self, key: str):
        data = self.get(f"model_{key}")
        if data is not None:
            return pickle.loads(data)
        return None


cache_manager = CacheManager()

# Глобальная память для синхронных callback'ов (UI-состояние)
GLOBAL_MEMORY = {
    'file_hash': None,
    'filename': None,
    'results': None,
}