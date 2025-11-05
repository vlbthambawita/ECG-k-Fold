__all__ = [
    "__version__",
    "hello_world",
]

__version__ = "0.1.1"


def hello_world(name: str = "World") -> str:
    """A simple hello world function to test the package.
    
    Args:
        name: Name to greet (default: "World")
        
    Returns:
        A greeting message
    """
    return f"Hello, {name}!"