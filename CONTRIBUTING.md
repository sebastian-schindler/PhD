# Contributing Guidelines

## Docstring Format

Use NumPy/SciPy style with minimal redundancy - avoid repeating information already in function signatures.

### Key Principles:
- **Omit type information** when already in function signature via type hints
- **Omit default values** when visible in signature
- **Omit "optional" designation** when defaults make it obvious
- **Include type information only when necessary** (e.g., "array-like", complex union types, duck typing scenarios, i.e. when the type hint doesn't capture the full contract)
- **Focus on behavior and usage** rather than redundant type information
- **Keep NumPy/SciPy section structure** with `Parameters`, `Returns`, etc.

### Template:
```python
from typing import Optional
from numpy.typing import ArrayLike

def function(
    param1: int, 
    param2: str = "default", 
    param3: Optional[ArrayLike] = None, 
    param4: Any = None, 
    **kwargs
) -> Any:
    """
    Brief description of the function.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1
        Description of param1 (type and default already in signature).
    param2
        Description of param2 (type and default already in signature).
    param3
        Description of param2 (type and default already in signature).
    param4 : function or object that behaves in a certain way
        Description (type specified in prose with just generic `Any` type in signature).
    kwargs
        Additional keyword arguments passed to underlying function.
    
    Returns
    -------
    result_name
        Description of the primary return value.
    additional_info
        Description of secondary return information.
    
    Examples
    --------
    >>> function(1, "test", [1, 2, 3], lambda x: x+1)
    result
    """
```

## Type Hinting Standards

Use specific, precise type hints rather than generic `Any` types.

### Key Principles:
- **Import typing utilities** like `Optional`, `Union`, `List`, `Tuple` as needed
- **Avoid `Any` types** - use specific types like `mpl.figure.Figure`, `callable`
- **Use string quotes for forward references** when importing optional dependencies (prevents import errors while maintaining type checking when packages are available)
- **Use Union types** for conditional returns based on parameters

### Examples:
```python
import matplotlib as mpl
import seaborn as sns
from typing import Union, List, Optional, tuple
from numpy.typing import ArrayLike

# Use of specific matplotlib types
def plot_function() -> mpl.figure.Figure:
    return plt.figure()

# Typing utilities
def get_axes() -> List[mpl.axes.Axes]:
    return [ax1, ax2, ax3]

# Tuples with specific types instead of generic Any
def get_plot_functions() -> tuple[mpl.axes.Axes, callable]:
    return main_axis, plot_func

# Optional dependencies that might not be installed (use string quotes)
def cluster_data(data: ArrayLike) -> "hdbscan.HDBSCAN":
    return hdbscan.HDBSCAN().fit(data)

def create_plot() -> Union[mpl.figure.Figure, "plotly.graph_objects.Figure"]:
    if use_plotly:
        return plotly_figure
    else:
        return matplotlib_figure

# Conditional returns based on parameters
def scan_function(return_range: bool = True) -> Union["awkward.Array", tuple["awkward.Array", tuple[int, int]]]:
    if return_range:
        return awkward_array, (min_value, max_value)
    else:
        return awkward_array
```

### Migration from Generic Types:
When updating existing code, replace generic types systematically:

- `Any` → specific type (e.g., `mpl.figure.Figure`, `callable`)
- `tuple[Any, Any]` → `tuple[specific_type1, specific_type2]`
- `list` → `List[specific_type]` when element type matters
- Generic return `Any` → actual return type from function analysis

### Dictionary Argument construct
Often-used construct for dictionary arguments (kwargs) that allow user input and provide defaults:
- Dict arguments to a function are never mutated but used to update the default dictionaries, defined early in function body
- For kwargs that are additional to the function's own kwargs, use `second_kwargs: dict = {}` instead of `second_kwargs: Optional[dict] = None`
- No need for None checks or initialization in function body
- This pattern is safe when the argument is later in the function only ever read from and never modified

Example:
```python
def plot_func(data, plot_kwargs: dict = {}):
    """Function with dict argument pattern."""
    plot_kwargs_ = {"color": "blue", "size": 10}
    plot_kwargs_.update(plot_kwargs)  # Safe since plot_kwargs is never modified
    plt.plot(data, **plot_kwargs_)  # Use defaults for plotting
```