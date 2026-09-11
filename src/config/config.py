import json
import os
from typing import Any, Dict

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Loads the JSON configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_active_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """Returns the configuration for the active model profile."""
    active = config.get("active_profile")
    if not active:
        raise ValueError("No active_profile specified in config.")
    
    profiles = config.get("profiles", {})
    if active not in profiles:
        raise ValueError(f"Active profile '{active}' not found in profiles.")
        
    return profiles[active]
