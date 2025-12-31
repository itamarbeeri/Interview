import os
import yaml

def get_video_path(root_dir=None):
    """
    Get the video path from the config file.
    Args:
        root_dir (str, optional): The root directory of the project. 
                                  If None, it's inferred from this file's location.
    Returns:
        str: The video path.
        None: If the config file is not found or the video is not found.
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        
    config_path = os.path.join(root_dir, "config.yml")
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    video_path = os.path.join(root_dir, config['video_dir_relative'], config['video_name'])
    
    # Check if video exists
    if not os.path.exists(video_path):
        return None
    
    return video_path

