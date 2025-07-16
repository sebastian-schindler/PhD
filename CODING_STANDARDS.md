# Coding Standards

## Docstring Format

Use NumPy/SciPy style structure but with minimal redundancy to avoid repeating information already present in function signatures.

### Key Principles:
- **Omit type information** when already in function signature via type hints
- **Omit default values** when visible in signature  
- **Omit "optional" designation** when defaults make it obvious
- **Include type information only when necessary** (e.g., "array-like", duck typing scenarios, complex union types)
- **Focus on behavior and usage** rather than redundant type information
- **Keep NumPy/SciPy section structure** with `Parameters`, `Returns`, etc.

### Template:
```python
def function(param1: int, param2: str = "default", param3):
    """
    Brief description of the function.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1
        Description of param1 (type and default already in signature).
    param2  
        Description of param2 (type and default already in signature).
    param3 : array-like
        Description of param3 (type specified because "array-like" can't be in signature).
    **kwargs
        Additional keyword arguments passed to underlying function.
    
    Returns
    -------
    Description of return value (type omitted if clear from signature annotation).
    
    Examples
    --------
    >>> function(1, "test", [1, 2, 3])
    result
    """
```

### When to Include Type Information:
- **Duck typing cases**: `array-like`, `file-like`, `dict-like`
- **Complex unions**: When type hints would be too verbose
- **Behavioral contracts**: When the type hint doesn't capture the full contract
- **Legacy code**: When adding type hints would be too disruptive

### When to Omit Type Information:
- **Simple types**: `int`, `str`, `bool`, `float` already in signature
- **Standard collections**: `list`, `dict`, `tuple` already in signature  
- **Clear from context**: When parameter name and description make type obvious
- **Optional parameters**: When default value makes type clear
